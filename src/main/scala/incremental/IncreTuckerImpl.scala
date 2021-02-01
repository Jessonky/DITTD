package incremental

import java.util.Date

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import paritions.Partitions._
import utils.Utils._

class IncreTuckerImpl(private var Mode: Int = 3,
                       private var ranks: Array[Int] = Array[Int](3, 3, 3),
                       private var mu: Double = 0.8,
                       private var cutNum: Int = 12,
                      private var partAlg: Int =0) {



  def run(fm0s: Array[BDM[Double]],
          g: Array[BDV[Double]],
          preDims: Array[Int],
          increTensor: RDD[linalg.Vector])
         (implicit sc: SparkContext): (Array[BDV[Double]], Array[BDM[Double]], Long, Double) = {

    val dims: Array[Int] = getDims(increTensor, Mode)

    val preFM0s: Array[BDM[Double]] = new Array[BDM[Double]](Mode)
    val preFM1s: Array[BDM[Double]] = new Array[BDM[Double]](Mode)
    val fm1s: Array[BDM[Double]] = new Array[BDM[Double]](Mode)
    for(i <- 0 until Mode){
      preFM0s(i) = fm0s(i).copy
      fm1s(i) = BDM.zeros(dims(i)-preDims(i), ranks(i))
    }

    val (c1s, c2s, c3) = cutTensor(increTensor, preDims, Mode) //RDD may be empty

    val dividedC1s = new Array[RDD[(Int, Vector)]](Mode)
    val dividedC2s = new Array[(RDD[(Int, Vector)], RDD[(Int, Vector)])](Mode)
    val dividedC3s = new Array[RDD[(Int, Vector)]](Mode)
    val groupedC1s = new Array[RDD[(Int, Iterable[Vector])]](Mode)
    val groupedC2s = new Array[(RDD[(Int, Iterable[Vector])], RDD[(Int, Iterable[Vector])])](Mode)
    val groupedC3s = new Array[RDD[(Int, Iterable[Vector])]](Mode)

    val startP = new Date().getTime
    for (i <- 0 until Mode) {
      val (j, k) = getOtherModesInorder(i)
      if(partAlg.equals(0)) {
        dividedC1s(i) = bucketPartition(c1s(i), cutNum, dims(i), i)
        dividedC2s(i) = (bucketPartition(c2s(j), cutNum, dims(i), i),
          bucketPartition(c2s(k), cutNum, dims(i), i))
        dividedC3s(i) = bucketPartition(c3, cutNum, dims(i), i)
      }
      else{
        dividedC1s(i) = greedyPartition(c1s(i), cutNum, dims(i) , i)
        dividedC2s(i) = (greedyPartition(c2s(j), cutNum, dims(i), i),
          greedyPartition(c2s(k), cutNum, dims(i), i))
        dividedC3s(i) = greedyPartition(c3, cutNum, dims(i), i)
      }
      groupedC1s(i) = dividedC1s(i).groupByKey(cutNum)
      groupedC2s(i) = (dividedC2s(i)._1.groupByKey(cutNum),
        dividedC2s(i)._2.groupByKey(cutNum))
      groupedC3s(i) = dividedC3s(i).groupByKey(cutNum)
    }
    val endP = new Date().getTime
    val partitionTime = endP - startP
    println("********** Partition Complete **********")

    val partVar = computeCV(groupedC1s, groupedC2s, groupedC3s)

    val broadcastPreDims: Broadcast[Array[Int]] = sc.broadcast(preDims)

    val broadcastDims: Broadcast[Array[Int]] = sc.broadcast(dims) // for baseline

    // C1s
    for (i <- 0 until Mode) {
      val (j, k) = getOtherModesInorder(i)
      // This is the baseline method
//      updateFM1(sc, i, g, ranks, fm0s(j), fm0s(k), dividedC1s(i),
//        broadcastPreDims, broadcastDims, cutNum, groupedC1s(i), fm1s(i))
      // This is the pro method
      updateFM1(sc, i, g, ranks, fm0s(j), fm0s(k), groupedC1s(i), fm1s(i), broadcastPreDims)

      modifiedGramSchmidt(fm1s(i))
    }

    // C2s
    for(i <- 0 until Mode){
      val (j, k) = getOtherModesInorder(i)
      preFM1s(i) = fm1s(i).copy
      // This is the baseline method
//      updateFM1(sc, i, g, ranks, fm0s(j), fm1s(k), dividedC2s(i)._1,
//        broadcastPreDims, broadcastDims, cutNum, groupedC2s(i)._1, fm1s(i))
      // This is the pro method
      updateFM1(sc, i, g, ranks, fm0s(j), fm1s(k), groupedC2s(i)._1, fm1s(i), broadcastPreDims)
      fm1s(i) = mergeFM1(preFM1s(i), fm1s(i), mu)
      modifiedGramSchmidt(fm1s(i))
      preFM1s(i) = fm1s(i).copy
      // This is the baseline method
//      updateFM1(sc, i, g, ranks, fm1s(j), fm0s(k), dividedC2s(i)._2,
//        broadcastPreDims, broadcastDims, cutNum, groupedC2s(i)._2, fm1s(i))
      // This is the pro method
      updateFM1(sc, i, g, ranks, fm1s(j), fm0s(k), groupedC2s(i)._2, fm1s(i), broadcastPreDims)
      fm1s(i) = mergeFM1(preFM1s(i), fm1s(i), mu)
      modifiedGramSchmidt(fm1s(i))
    }

    //C3s
    for(i <- 0 until Mode){
      val (j, k) = getOtherModesInorder(i)
      preFM1s(i) = fm1s(i).copy
      // This is the baseline method
//      updateFM1(sc, i, g, ranks, fm1s(j), fm1s(k), dividedC3s(i),
//        broadcastPreDims, broadcastDims, cutNum, groupedC3s(i), fm1s(i))
      // This is the pro method
      updateFM1(sc, i, g, ranks, fm1s(j), fm1s(k), groupedC3s(i), fm1s(i), broadcastPreDims)
      fm1s(i) = mergeFM1(preFM1s(i), fm1s(i), mu)
      modifiedGramSchmidt(fm1s(i))
    }

    println("********** UpdateFM1s Complete **********")


    // MGS
    val fmts: Array[BDM[Double]] = new Array[BDM[Double]](Mode)
    for(i <- 0 until Mode){
      fmts(i) = BDM.vertcat(fm0s(i), fm1s(i))
      modifiedGramSchmidt(fmts(i))
      fm0s(i) = fmts(i)(0 until preDims(i), ::)
      fm1s(i) = fmts(i)(preDims(i) until dims(i), ::)
    }


//     updateG0 X-111
    for(i<-0 until Mode){
      updateG0(g, ranks, fm0s(i).t * preFM0s(i), i)
    }

    // C1s X-211\121\112
    for(i <- 0 until Mode){
      val (j, k) = getOtherModesInorder(i)
      updateG(sc, i, fm1s(i), fm0s(j), fm0s(k), groupedC1s(i), broadcastPreDims, g, ranks)
    }

    // C2s X-122\212\221
    for(i <- 0 until Mode){
      val (j, k) = getOtherModesInorder(i)
      updateG(sc, i, fm1s(i), fm0s(j), fm1s(k), groupedC2s(i)._1, broadcastPreDims, g, ranks)
    }

    // C3s X-222
    for(i <- 0 until Mode){
      val (j, k) = getOtherModesInorder(i)
      updateG(sc, i, fm1s(i), fm1s(j), fm1s(k), groupedC3s(i), broadcastPreDims, g, ranks)
    }

    println("********** UpdateG Complete **********")

    (g, fm1s, partitionTime, partVar)
  }


  def setMode(m: Int): this.type = {
    require(m > 0, s"Mode of the tenor must be positive but got $m ")
    Mode = m
    this
  }

  def setRanks(r: Array[Int]): this.type = {
    require(r != null, s"The ranks shouldn't be null!")
    ranks = r
    this
  }

  def setMu(m: Double): this.type = {
    require(m > 0, s"The forgetting coefficient must be positive but got $m ")
    mu = m
    this
  }

  def setCutNum(c: Int): this.type = {
    require(c > 0, s"The cutNum of tenor must be positive but got $c ")
    cutNum = c
    this
  }

  def setPartAlg(a: Int): this.type = {
    require(a.equals(0) || a.equals(1), s"partitioning algorithm a must be 0 or 1, but got $a ")
    partAlg = a
    this
  }

}

