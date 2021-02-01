package paritions

import java.util.{Comparator, PriorityQueue}

import breeze.numerics.sqrt
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import utils.Utils

/**
 * Partition related methods.
 */
object Partitions {

  /**
   * compute the average CV of all the partitioned sub-tensors
   * @param groupedC1s the partitioned sub-tensors in C1
   * @param groupedC2s the partitioned sub-tensors in C2
   * @param groupedC3s the partitioned sub-tensors in C3
   * @return
   */
  def computeCV(groupedC1s: Array[RDD[(Int, Iterable[Vector])]],
                      groupedC2s: Array[(RDD[(Int, Iterable[Vector])], RDD[(Int, Iterable[Vector])])],
                      groupedC3s: Array[RDD[(Int, Iterable[Vector])]]): Double = {
    var partCV = 0.0
    val m = groupedC1s.length
    var den = groupedC1s.length * 4
    for (i <- 0 until m) {
      val v1 = computeCV(groupedC1s(i))
      val v2 = computeCV(groupedC2s(i)._1)
      val v3 = computeCV(groupedC2s(i)._2)
      val v4 = computeCV(groupedC3s(i))
      if(v1<0) den -= 1 //RDD is empty
      else partCV += v1
      if(v2<0) den -= 1
      else partCV += v2
      if(v3<0) den -= 1
      else partCV += v3
      if(v4<0) den -= 1
      else partCV += v4
    }
    partCV /= den
    partCV
  }

  /**
   * compute the CV of the given partitioned tensor
   * @param groupedTensor the given partitioned tensor
   * @return
   */
  def computeCV(groupedTensor: RDD[(Int, Iterable[Vector])]): Double = {

    val arr: Array[Int] = groupedTensor.mapValues(x => x.count(_ != null)).values.collect()

    if(arr.length.equals(0)){
      return -1.0
    }//RDD is empty

    var sum: Long = 0
    arr.foreach { x =>
      sum += x
    }
    val mean = sum / arr.length

    var sum2: Long = 0
    arr.foreach { x =>
      sum2 += (x - mean) * (x - mean)
    }
    val variance = sum2 / arr.length

    val result = sqrt(variance) / mean
    result
  }


  /**
   * compute the number of non-zero elements (nnz)
   * in every slice of the given tensor in the given mode
   *
   * @param tensorData the given tensor
   * @param mode the mode
   * @param dim the dim of the mode
   * @return  [(index, value)] index represents dim index; value represents nnz
   */
  def getNonZeros4mX(tensorData: RDD[Vector], mode: Int, dim: Int): Array[Int] = {
    val tmpList: Seq[(Int, Long)] = tensorData.map(x => (x.apply(mode).toInt, 1)).countByKey().toList
    val nonZeros4mX = new Array[Int](dim + 1)
    tmpList.foreach {
      x => nonZeros4mX(x._1) = x._2.toInt
    }
    nonZeros4mX
  }

  /**
   * greedy partitioning according to the nnz in every slice
   *
   * @param nonZeros4mX [(index, value)] index represents dim index; value represents nnz
   * @param dim         the dim
   * @param tensorData  the tensor
   * @param cutSize     the partition number
   * @return [(index, value)] index represents dim index; value represents partition index
   */
  def getId4mGreedy(nonZeros4mX: Array[Int],
                    dim: Int,
                    tensorData: RDD[Vector],
                    cutSize: Int): Array[Int] = {
    val totalLines = Utils.getTotalLines(tensorData)

    val threshold = totalLines / cutSize // optimal size
    val id4mX = new Array[Int](dim + 1)

    var index = 1
    var pre = 0
    var cur = 0
    for (i <- nonZeros4mX.indices) {
      cur += nonZeros4mX(i)
      if (cur <= threshold) {
        id4mX(i) = index
      }
      else {
        if ((cur - threshold) > (threshold - pre)) {
          id4mX(i) = if (index == cutSize) index else index + 1
          cur = nonZeros4mX(i)
        }
        else {
          id4mX(i) = index
          cur = 0
        }
        index = if (index == cutSize) index else index + 1
      }
      pre = cur
    }
    id4mX
  }


  /**
   * greedy partitioning according to the nnz in every slice
   *
   * @param nonZeros4mX [(index, value)] index represents dim index; value represents nnz
   * @param dim         the dim
   * @param cutSize     the partition number
   * @return [(index, value)] index represents dim index; value represents partition index
   */
  def getId4mBucket(nonZeros4mX: Array[Int],
                    dim: Int,
                    cutSize: Int): Array[Int] = {
    val id4mX = new Array[Int](dim + 1) // 每个位置上的值表示当前的id
    val sorted: Array[(Int, Int)] = nonZeros4mX.zipWithIndex.sortWith((x, y) => x._1 > y._1) // 从大到小排序

    //(key, value) key:partition index, value: nnz in the partition
    val minHeap = new PriorityQueue[(Int, Int)](cutSize, new Comparator[(Int, Int)]() {
      override def compare(o1: (Int, Int), o2: (Int, Int)): Int = {
        o1._2 - o2._2
      }
    })
    for (i <- 1 to cutSize) {
      minHeap.offer(i, 0)
    }

    sorted.foreach { x =>
      val curPeek = minHeap.poll()
      id4mX(x._2) = curPeek._1
      minHeap.offer(curPeek._1, curPeek._2 + x._1)
    }

    id4mX
  }


  /**
   * greedy partitioning for the given tensor
   *
   * @param tensorData tensor
   * @param cutNum partition number
   * @param dim  the dim of the given mode
   * @param mode the mode
   * @param sc SparkContext
   * @return RDD[(Int, Vector)] (partID, tensor)
   */
  def greedyPartition(tensorData: RDD[Vector],
                      cutNum: Int,
                      dim: Int,
                      mode: Int)
                     (implicit sc: SparkContext): RDD[(Int, Vector)] = {
    if(tensorData.isEmpty()) {
      val emptyTensor = tensorData.map(x=>( -1, x))
      return emptyTensor
    }
    val nonZeros4m: Array[Int] = getNonZeros4mX(tensorData, mode, dim)
    val id4m: Array[Int] = getId4mGreedy(nonZeros4m, dim, tensorData, cutNum)

    val broadId: Broadcast[Array[Int]] = sc.broadcast(id4m)
    val dividedTensor: RDD[(Int, Vector)] = tensorData.map {
      x =>
        (broadId.value(x.apply(mode).toInt),
          x)
    }
    dividedTensor
  }

  /**
   * bucket partitioning for the given tensor
   *
   * @param tensorData tensor
   * @param cutNum partition number
   * @param dim  the dim of the given mode
   * @param mode the mode
   * @param sc SparkContext
   * @return RDD[(Int, Vector)] (partID, tensor)
   */
  def bucketPartition(tensorData: RDD[Vector],
                      cutNum: Int,
                      dim: Int,
                      mode: Int)
                     (implicit sc: SparkContext): RDD[(Int, Vector)] = {
    if(tensorData.isEmpty()) {
      val emptyTensor = tensorData.map(x=>( -1, x))
      return emptyTensor
    }
    val nonZeros4m: Array[Int] = getNonZeros4mX(tensorData, mode, dim )
    val id4m: Array[Int] = getId4mBucket(nonZeros4m, dim, cutNum)

    val broadId = sc.broadcast(id4m) // 广播id数组到每个节点
    val dividedTensor: RDD[(Int, Vector)] = tensorData.map {
      x =>
        (broadId.value(x.apply(mode).toInt),
          x)
    }

    dividedTensor
  }

}