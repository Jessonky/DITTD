package utils

import java.io.{File, PrintWriter}

import breeze.linalg.{norm, pinv, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.sqrt
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * The operations related with distributed incremental tucker decomposition
 */
object Utils {

  /**
   * update core tensor G according to the factor matrices and incremental tensor
   * @param sc SparkContext
   * @param i the mode of fm0
   * @param fm0 factor matrix 0
   * @param fm1 factor matrix 1
   * @param fm2 factor matrix 2
   * @param groupedTensor the related incremental tensor
   * @param broadcastPreDims the pre dims
   * @param g the pre G
   * @param ranks the rank of tucker decomposition
   */
  def updateG(sc: SparkContext, i: Int,
              fm0: BDM[Double], fm1: BDM[Double], fm2: BDM[Double],
              groupedTensor: RDD[(Int, Iterable[Vector])],
              broadcastPreDims: Broadcast[Array[Int]],
              g: Array[BDV[Double]], ranks: Array[Int]): Unit = {
    val (j, k) = getOtherModesInorder(i)
    val curMode = Array(i,j,k)
    val broadcastCurMode: Broadcast[Array[Int]] = sc.broadcast(curMode)
    val broadcastFM1 = sc.broadcast(fm1)
    val broadcastFM2 = sc.broadcast(fm2)
    val TMM: Array[(Int, BDV[Double])] = computeTMM(groupedTensor, broadcastPreDims,
        broadcastCurMode, broadcastFM1, broadcastFM2).collect()
    val tmm: BDM[Double] = IndexedBDV2BDM(fm0.rows, fm1.cols*fm2.cols, TMM)
    val gm = fm0.t * tmm
    matrixAddTensor(g, gm, ranks, i)
  }

  /**
   * transfer Indexed BDV to BDM
   * @param rowNum the row number
   * @param colNum the col number
   * @param indexedBDVs the indexed BDV
   * @return BDM
   */
  def IndexedBDV2BDM(rowNum: Int, colNum: Int, indexedBDVs: Array[(Int, BDV[Double])]): BDM[Double] ={
    val M: BDM[Double] = BDM.zeros(rowNum, colNum)
    indexedBDVs.foreach( tuple =>{
      val index = tuple._1
      val row = tuple._2.t
      M(index, ::) := row
    } )
    M
  }

  /**
   * Modified Gram Schmidt Processing
   * Orthonormalizing the matrix
   * @param A matrix
   */
  def modifiedGramSchmidt(A: BDM[Double]): Unit = {
    val cols = A.cols
    for(i <- 0 until cols){
      A(::,i) :*= 1.0 / norm(A(::,i))
      for(j <- i+1 until cols){
        A(::, j) := A(::, j) - (A(::, i) dot A(::, j)) * A(::, i)
      }
    }
  }

  def mergeFM1(preM: BDM[Double], currM: BDM[Double], mu: Double): BDM[Double] ={
    preM :*= mu
    currM :*= (1-mu)
    preM + currM
  }

  /**
   * transfer matrix to tensor
   * @param g the tensor to be transfered to
   * @param gm the matrix to be transfered
   * @param ranks the modes of g
   * @param m the matricized mode
   */
  def matrix2Tensor(g: Array[BDV[Double]], gm: BDM[Double], ranks: Array[Int], m: Int): Unit = {
    val (m1, m2) = getOtherModesInorder(m)
    val coordinates = new Array[Int](3)
    for(i <- 0 until gm.rows)
      for(j <- 0 until gm.cols){
        coordinates(m) = i
        coordinates(m1) = j % ranks(m1)
        coordinates(m2) = j / ranks(m1)
        val order = coordinates(0)*ranks(1)*ranks(2) + coordinates(1)*ranks(2) + coordinates(2)
        g(order)(3) = gm(i,j)
      }
  }

  /**
   * add a tensor and a matricized tensor
   * @param g the curr tensor
   * @param gm the matricized tensor
   * @param ranks the modes of g
   * @param m the matricized mode
   */
  def matrixAddTensor(g: Array[BDV[Double]], gm: BDM[Double], ranks: Array[Int], m: Int): Unit = {
    val (m1, m2) = getOtherModesInorder(m)
    val coordinates = new Array[Int](3)
    for(i <- 0 until gm.rows)
      for(j <- 0 until gm.cols){
        coordinates(m) = i
        coordinates(m1) = j % ranks(m1)
        coordinates(m2) = j / ranks(m1)
        val order = coordinates(0)*ranks(1)*ranks(2) + coordinates(1)*ranks(2) + coordinates(2)
        g(order)(3) += gm(i,j)
      }
  }

  /**
   * update core tensor G
   * @param g the pre and curr G
   * @param ranks the modes of core tensor G
   * @param M the matrix
   * @param m the mode
   */
  def updateG0(g: Array[BDV[Double]], ranks: Array[Int], M: BDM[Double], m: Int): Unit = {
    val gm: Matrix = tensorMatricization(g, ranks, m)
    val rm = M * new BDM(gm.numRows, gm.numCols, gm.toArray)
    matrix2Tensor(g, rm, ranks, m)
  }

  /**
   * update factor matrix
   * @param sc SparkContext
   * @param i mode
   * @param g core tensor
   * @param ranks ranks of core tensor
   * @param fm1 the first factor matrix
   * @param fm2 the second factor matrix
   * @param groupedTensor partitioned tensor
   * @param toUpdateFM1 the factor matrix to be updated
   * @param broadcastPreDims the broadcast preDims
   */
  def updateFM1(sc: SparkContext,
                i: Int, g: Array[BDV[Double]], ranks: Array[Int],
                fm1: BDM[Double],
                fm2: BDM[Double],
                groupedTensor: RDD[(Int, Iterable[Vector])],
                toUpdateFM1: BDM[Double],
                broadcastPreDims: Broadcast[Array[Int]]): Unit ={
    val (j, k) = getOtherModesInorder(i)
    val curMode = Array(i,j,k)
    val broadcastCurMode: Broadcast[Array[Int]] = sc.broadcast(curMode)

    //g(i) the i-mode matricization of tensor g
    val gm: BDM[Double] = tensor2BDM(g, ranks, i)

    //****
    // compute g(i).t * (g(i) ((fm2.t * fm2) \kdot (fm1.t * fm1)) g(i).t)^+
    // if fm is column orthogonal, fm.t* fm = I, which could be avoid the gram computation
//    val fm1Gram = fm1.t * fm1
//    val fm2Gram = fm2.t * fm2
//    val fmkfm:BDM[Double] = kroneckerProduct(fm2Gram, fm1Gram)
//    val gmmGram = gm*fmkfm*gm.t
//    val gmmGramPlus = pinv(gmmGram)
//    val gTgmmGramPlus: BDM[Double] = gm.t * gmmGramPlus
//    val broadcastRight = sc.broadcast(gTgmmGramPlus)
    //****

    val right = pinv(gm)
    val broadcastRight = sc.broadcast(right)

    //****
    //compute x(i) * (fm2.t \kdot fm1) * right row-wisely
    val broadcastFM1 = sc.broadcast(fm1)
    val broadcastFM2 = sc.broadcast(fm2)
    val resultRDD: RDD[(Int, BDV[Double])] =
      computeTMM(groupedTensor, broadcastPreDims, broadcastCurMode, broadcastFM1, broadcastFM2)
        .map(x => (x._1, broadcastRight.value.t * x._2))
    //****

    resultRDD.collect().foreach(x => {
      toUpdateFM1(x._1, ::) := x._2.t
    })
  }


  /**
   * compute tensor * (FM2 \otimes FM1)
   * @param groupedTensor the partitioned tensor
   * @param broadcastPreDims the broadcast pre dims
   * @param broadcastCurMode the broadcast curr mode order
   * @param broadcastFM1 the corresponding factor matrix 1
   * @param broadcastFM2 the corresponding factor matrix 2
   * @return Indexed BDV for tensor * (FM2 \otimes FM1) result
   */
  private def computeTMM(groupedTensor: RDD[(Int, Iterable[Vector])],
                         broadcastPreDims: Broadcast[Array[Int]],
                         broadcastCurMode: Broadcast[Array[Int]],
                         broadcastFM1: Broadcast[BDM[Double]],
                         broadcastFM2: Broadcast[BDM[Double]]): RDD[(Int, BDV[Double])] = {
    groupedTensor.values.flatMap(x => x).map(x => {
      val i = getFMIndex(broadcastCurMode.value(0), broadcastPreDims.value(broadcastCurMode.value(0)), x)
      val j = getFMIndex(broadcastCurMode.value(1), broadcastPreDims.value(broadcastCurMode.value(1)), x)
      val k = getFMIndex(broadcastCurMode.value(2), broadcastPreDims.value(broadcastCurMode.value(2)), x)
      val row1 = broadcastFM1.value(j, ::).inner
      val row2 = broadcastFM2.value(k, ::).inner
      val Q = row1.length
      val R = row2.length
      val kRow: BDV[Double] = BDV.zeros[Double](Q * R)
      for (it <- 0 until R) {
        kRow(it * Q until (it + 1) * Q) := row2(it) * row1
      }
      val currRow = x(3) * kRow
      (i, currRow)
    }).reduceByKey(_+_)
  }

  /**
   * get the corresponding indexed for the mode of tensor element
   * @param m the mode
   * @param preDimsM the dims of the pre tensor in this mode
   * @param x the tensor element
   * @return index (start from 0 for matrix, start from 1 for tensor)
   */
  private def getFMIndex(m: Int, preDimsM: Int, x: Vector): Int = {
    if (x.apply(m).toInt > preDimsM)
      x.apply(m).toInt - 1 - preDimsM
    else
      x.apply(m).toInt - 1
//      x.apply(m).toInt - 1 - 1 //-1 is enough, there maybe a bug
  }

  /**
   * the base update factor matrix method
   * @param sc SparkContext
   * @param i mode
   * @param g core tensor
   * @param ranks ranks of core tensor
   * @param fm1 the first factor matrix
   * @param fm2 the second factor matrix
   * @param dividedTensor the partitioned tensor in lines
   * @param broadcastPreDims the broadcast preDims
   * @param cutNum the partition number
   * @param groupedTensor the partitioned tensor in groups
   * @param toUpdateFM1 the factor matrix to be updated
   */
  def updateFM1(sc: SparkContext,
                i: Int, g: Array[BDV[Double]], ranks: Array[Int],
                fm1: BDM[Double], fm2: BDM[Double],
                dividedTensor: RDD[(Int, Vector)],
                broadcastPreDims: Broadcast[Array[Int]],
                broadcastDims: Broadcast[Array[Int]],
                cutNum: Int, groupedTensor: RDD[(Int, Iterable[Vector])],
                toUpdateFM1: BDM[Double]): Unit ={

    if(fm1.rows==0 || fm2.rows ==0){
      for( i <- 0 until toUpdateFM1.rows){
        toUpdateFM1(i,::):= 0.0
      }
      return
    }

    val (j, k) = getOtherModesInorder(i)
    val curMode = Array(i,j,k)
    val broadcastCurMode: Broadcast[Array[Int]] = sc.broadcast(curMode)

    val gmmt = getGMMT(sc, i, g, ranks, fm1, fm2)
    val gmmGram = gmmt.computeGramianMatrix()
    // ********
    val gmmGramPlus = Matrices.dense(gmmGram.numRows, gmmGram.numRows,
      pinv(new BDM(gmmGram.numRows, gmmGram.numRows, gmmGram.toArray)).toArray)
    val gmmPlus: IndexedRowMatrix = gmmt.multiply(gmmGramPlus)
    // It is unnecessary to get all rows of gmmPlus, since X may be zero.
    //********

    val indexedRowFM1 = tensorMultiIndexedRowMatrix(dividedTensor,
      broadcastCurMode, broadcastPreDims, broadcastDims, gmmPlus, cutNum, groupedTensor)

    indexedRowFM1.collect().foreach(x => {
      toUpdateFM1(x._1, ::) := x._2.t
    })
  }

  /**
   * tensor * indexedRowMatrix
   * @param dividedTensor
   * @param broadcastCurMode
   * @param broadcastPreDims
   * @param irm
   * @param cutNum
   * @param groupedTensor
   * @return
   */
  def tensorMultiIndexedRowMatrix(dividedTensor: RDD[(Int, Vector)],
                                  broadcastCurMode: Broadcast[Array[Int]],
                                  broadcastPreDims: Broadcast[Array[Int]],
                                  broadcastDims: Broadcast[Array[Int]],
                                  irm: IndexedRowMatrix,
                                  cutNum: Int,
                                  groupedTensor: RDD[(Int, Iterable[Vector])]): RDD[(Int, BDV[Double])] ={
    //(indexJ, PartID)
    val indexJ2Part: RDD[(Long, Int)] = dividedTensor.map(x=>{
      val partID = x._1
      val tensor = x._2
      val indexJ = getMatrizationIndexJ(tensor, broadcastCurMode, broadcastPreDims, broadcastDims)
      (indexJ, partID)
    })

    // (indexJ, Vector)
    val indexedGmmPlus = irm.rows.map(x=> (x.index, x.vector))


    // (PartID, indexedRow)
    val dividedGmmPlus = indexJ2Part.join(indexedGmmPlus).map(x=>(x._2._1, IndexedRow(x._1, x._2._2)))

    val partGmmPlus = dividedGmmPlus.groupByKey(cutNum)

    val fm1Split = groupedTensor.join(partGmmPlus).values.map(x =>{
      //x: [x], [g]
      val xList = x._1
      val gmmPlusRows: Iterable[IndexedRow] = x._2
      val gmmPlusRowsMap = gmmPlusRows.map(x=>(x.index, x.vector)).toMap

      val tmp: Iterable[(Int, BDV[Double])] = xList.map(x=>{
        val indexJ = getMatrizationIndexJ(x, broadcastCurMode, broadcastPreDims, broadcastDims)
        val indexI = getFMIndex(broadcastCurMode.value(0), broadcastPreDims.value(broadcastCurMode.value(0)), x)
        val curRow: Vector = gmmPlusRowsMap(indexJ)
        val tmpR = new BDV(curRow.toArray)
        tmpR :*= x.apply(3)
        (indexI, tmpR)
      })
      tmp

    }).flatMap(x=>x)

//    fm1Split.reduceByKey((row1, row2) => addVector(row1, row2)).collect()
    fm1Split.reduceByKey((row1, row2) => row1 + row2)

  }


  /**
   * tranfor BDM to Indexed RDD
   * @param sc SparkContext
   * @param M BDM[Double]
   * @return
   */
  def tranforBDMtoIndexedRDD(sc: SparkContext, M: BDM[Double]): RDD[(Long, Array[Double])] ={
    val I = M.rows
    val J = M.cols
    val Mrows = new Array[(Long, Array[Double])](I)

    for(i <- 0 until I){
      val tmp: Array[Double] = new Array[Double](J)
      for(j <- 0 until J){
        tmp(j) = M(i,j)
      }
      Mrows(i) = (i, tmp)
    }
    sc.makeRDD(Mrows)
  }

  /**
   * Indexed vectors to BDM (the matrix is known)
   * @param M the matrix
   * @param indexedVectors the indexed vectors
   */
  def IndexedVec2BDM(M: BDM[Double], indexedVectors: Array[(Int, Vector)]): Unit ={
    indexedVectors.foreach( tuple =>{
      val index = tuple._1
      val row = tuple._2
      M(index, ::) := BDV(row.toArray).t
    } )
  }

  //add vector
  def addVector(row1: Vector, row2: Vector): Vector ={
    val a1 = row1.toArray
    val a2 = row2.toArray
    for(i <- a1.indices){
      a1(i) += a2(i)
    }
    Vectors.dense(a1)
  }

  /**
   * generate big kroneckerProduct in IndexedRowMatrix
   * @param sc SparkContext
   * @param A matrix A
   * @param B matirx B
   * @return
   */
  def kroneckerProduct(sc: SparkContext,A: BDM[Double], B: BDM[Double]): IndexedRowMatrix = {

    val ARDD: RDD[(Long, Array[Double])] = tranforBDMtoIndexedRDD(sc, A)
    val BRDD: RDD[(Long, Array[Double])] = tranforBDMtoIndexedRDD(sc, B)

    val I2 = B.rows

    val broadcastI2 = sc.broadcast(I2)


    val AkB = ARDD.cartesian(BRDD).map(R => {
      val row1 = R._1
      val row2 = R._2

      val R1 = row1._2.length
      val R2 = row2._2.length
      val resultArr = new Array[Double](R1*R2)

      for(i <- row1._2.indices)
        for(j <- row2._2.indices){
          resultArr(i*R2 + j) = row1._2(i) * row2._2(j)
        }

      val I2 = broadcastI2.value

      IndexedRow(row1._1 * I2.toLong + row2._1, Vectors.dense(resultArr))
    })
      //.sortBy(x=>x.index)  // Sort is expensive, is it necessary?

    new IndexedRowMatrix(AkB)
  }

  /**
   * generate kroneckerProduct in BDM
   * @param A matrix A
   * @param B matrix B
   * @return
   */
  def kroneckerProduct(A: BDM[Double], B: BDM[Double]): BDM[Double] = {
      val I = A.rows
      val J = A.cols
      val K = B.rows
      val L = B.cols
      val x = BDM.zeros[Double](I*K, J*L)
      for(i <- 0 until I)
        for(j <- 0 until J){
          x( (i * K) until ((i + 1) * K), (j * L) until ((j + 1) * L)) :=  A(i,j) * B
        }
      x
  }

  /** get the index J for tensor matrization*/
  def getMatrizationIndexJ(tensor: Vector,
                           broadcastCurMode: Broadcast[Array[Int]],
                           broadcastPreDims: Broadcast[Array[Int]],
                           broadcastDims: Broadcast[Array[Int]]): Long ={
    val j = getFMIndex(broadcastCurMode.value(1),
      broadcastPreDims.value(broadcastCurMode.value(1)), tensor)
    val k = getFMIndex(broadcastCurMode.value(2),
      broadcastPreDims.value(broadcastCurMode.value(2)), tensor)

//    k + j*J

    val dimJ = if (tensor.apply(broadcastCurMode.value(1)) >
      broadcastPreDims.value(broadcastCurMode.value(1)))
      broadcastDims.value(broadcastCurMode.value(1)) - broadcastPreDims.value(broadcastCurMode.value(1))
    else  broadcastPreDims.value(broadcastCurMode.value(1))

    k.toLong + j*dimJ.toLong

  }

  def tensorMatricization(g: Array[BDV[Double]], ranks:Array[Int], n: Int): Matrix = {
    val (n1, n2) = getOtherModesInorder(n)
    val I = ranks(n)
    val J = ranks(n1)*ranks(n2)
    val G = BDM.zeros[Double](I,J)
    g.foreach(x => {
      G(x(n).toInt - 1, ((x(n1) - 1) + (x(n2)-1) * ranks(n1)).toInt) = x(3)
    })
    Matrices.dense(I,J, G.toArray)
  }

  def tensor2BDM(g: Array[BDV[Double]], ranks:Array[Int], n: Int): BDM[Double] = {

    val (n1, n2) = getOtherModesInorder(n)
    val I = ranks(n)
    val J = ranks(n1)*ranks(n2)
    val G = BDM.zeros[Double](I,J)

    g.foreach(x => {
      G(x(n).toInt - 1, ((x(n1) - 1) + (x(n2)-1) * ranks(n1)).toInt) = x(3)
    })

    G
  }

  //compute (G * (m2 \otimes m1))^t
  def getGMMT(sc: SparkContext, m: Int, g: Array[BDV[Double]], ranks: Array[Int],
             m1: BDM[Double], m2: BDM[Double]): IndexedRowMatrix ={
    
    val kp: IndexedRowMatrix = kroneckerProduct(sc, m2, m1)
    val gm: Matrix = tensorMatricization(g, ranks, m)

    kp.multiply(gm.transpose)
  }

  /**
   * get the dims of the given tensor
   */
  def getDims(tensorData: RDD[Vector], Mode: Int): Array[Int] = {
    val sizeVector: Vector = new RowMatrix(tensorData).computeColumnSummaryStatistics().max
    val dims = new Array[Int](Mode)
    for (i <- 0 until Mode) {
      dims(i) = sizeVector(i).toInt
    }
    dims
  }

  /**
   * get the CV for the given tensor
   * @param tensorData
   * @param Mode
   * @return
   */
  def getnnzCV(tensorData: RDD[Vector], Mode: Int): Double = {
    var cv = 0.0
    for (i <- 0 until Mode) {
      val tmpArray: Array[Long] = tensorData.map(x => (x.apply(i).toInt, 1)).countByKey().values.toArray
      val mean = 1.0 * tmpArray.sum / tmpArray.size
      var sum = 0.0
      tmpArray.foreach( x => sum += (x - mean)*(x - mean))
      val variance = sum / tmpArray.length
      val result = sqrt(variance) / mean
      println(result)
      cv += result
    }
    cv/Mode
  }


  def initialCoordinatesG(ranks: Array[Int]): Array[BDV[Double]] ={
    val len = ranks.product
    val g = new Array[BDV[Double]](len)
    var count = 0
    for(i <- 1 to ranks(0)){
      for(j <- 1 to ranks(1)){
        for(k <- 1 to ranks(2)){
          g(count) = BDV(i.toDouble,j.toDouble,k.toDouble, Random.nextDouble())
          count += 1
        }
      }
    }
    g
  }

  /**
   * cut tensor to tensor categories
   */
  def cutTensor(increTensor: RDD[Vector],
                preDims: Array[Int],
                Mode: Int): (Array[RDD[Vector]], Array[RDD[Vector]], RDD[Vector]) = {
    val c3 = increTensor.filter { x =>
      x.apply(0) > preDims(0) && x.apply(1) > preDims(1) && x.apply(2) > preDims(2)
    }//222
    val c1s = new Array[RDD[Vector]](Mode)
    val c2s = new Array[RDD[Vector]](Mode)
    for (i <- c1s.indices) {
      val j = (i + 1) % 3
      val k = (i + 2) % 3
      c1s(i) = increTensor.filter { x =>
        x.apply(i) > preDims(i) && x.apply(j) <= preDims(j) && x.apply(k) <= preDims(k)
      }// 211 121 112
      c2s(i) = increTensor.filter { x =>
        x.apply(i) <= preDims(i) && x.apply(j) > preDims(j) && x.apply(k) > preDims(k)
      }// 122 212 221
    }
    (c1s, c2s, c3)
  }

  /**
   * get line sets for factor matrix
   */
  def getLineSets(tensorData: RDD[Vector],
                  dims: Array[Int],
                  Mode: Int): Array[Array[Int]] = {
    val coll: Array[Array[Int]] = tensorData.map { x =>
      Array(x.apply(0).toInt, x.apply(1).toInt, x.apply(2).toInt)
    }.collect()

    val lineSets = new Array[Array[Int]](Mode)
    for (m <- 0 until Mode) {
      lineSets(m) = new Array[Int](dims(m) + 1)
      coll.foreach { x =>
        if (x.apply(m) <= dims(m)) {
          lineSets(m)(x.apply(m)) = 1
        }
      }
    }

    lineSets
  }


  /**
   * get the total number of tensor
   */
  def getTotalLines(tensorData: RDD[linalg.Vector]): Long = {
    new RowMatrix(tensorData).numRows()
  }


  // get other modes in order
  def getOtherModesInorder(mode: Int): (Int, Int) = {
    val i1: Int = (mode + 1) % 3
    val i2: Int = (mode + 2) % 3
    if(i1 < i2)
      (i1, i2)
    else
      (i2, i1)
  }


  /**
   * group tensor by partID
   */
  def groupTensorByLayer(dividedTensor: RDD[(Int, Vector)],
                         cutNum: Int): RDD[(Int, Iterable[Vector])] = {

    dividedTensor.groupByKey(cutNum) // 按照层数分配partition
  }

  /**
   * read tensorRDD from file
   */
  def fromFile(path: String)(implicit sc: SparkContext): RDD[Vector] = {
    sc.textFile(path).map(s => Vectors.dense(s.split(',').map(_.toDouble)))
  }


  /**
   * filter factor matrix according to line set
   */
  def filterFMs(fms: Array[IndexedRowMatrix],
                broadLineSets: Broadcast[Array[Array[Int]]],
                Mode: Int): Array[IndexedRowMatrix] = {
    val filteredFMs = new Array[IndexedRowMatrix](Mode)

    for (i <- fms.indices) {
      val filteredRows = fms(i).rows.filter(x => broadLineSets.value(i)(x.index.toInt) == 1)
      filteredFMs(i) = new IndexedRowMatrix(filteredRows)
    }

    filteredFMs
  }


  /**
   * broadcost indexedRowMatrix in map(Long, BDV[Double)
   */
  def broadFM(fms: Array[IndexedRowMatrix])
             (implicit sc: SparkContext): Array[Broadcast[Map[Long, BDV[Double]]]] = {
    val broadFMs = new Array[Broadcast[Map[Long, BDV[Double]]]](fms.length)
    for (i <- broadFMs.indices) {
      broadFMs(i) = sc.broadcast(IRMtoMap(fms(i)))
    }
    broadFMs
  }


  /**
   * compute x * g * U1 * U2
   * This is a wrong method
   */
  def computeG(x: Double,
               u1: BDV[Double],
               u2: BDV[Double],
               g: Broadcast[Array[Array[Array[Double]]]],
               m: Int): BDV[Double] = {
    val (m1, m2) = getOtherModes(m)

    val arr1 = Array.ofDim[Double](m, m2) // 2 dimensional array
    for (i <- 0 to m) {
      for (k <- 0 to m2) {
        val tmpArr = Array[Double](m1)
        for (j <- 0 to m1) {
          tmpArr(j) = g.value(i)(j)(k)
        }
        arr1(i)(k) = u1.t * BDV(tmpArr)
      }
    }

    val arr2 = Array.ofDim[Double](m)
    for (i <- 0 to m) {
      arr2(i) = u2.t * BDV(arr1(i))
    }

    val gg: BDV[Double] = BDV(arr2)
    val mm: BDM[Double] = gg * gg.t


    (gg.t * pinv(mm)).t :*= x
  }


  /**
   * initial G: Array[Array[Array[Double]]]
   */
  def initialG(ranks: Array[Int]): Array[Array[Array[Double]]] = {
    val g = Array.ofDim[Double](ranks(0), ranks(1), ranks(2))
    for (i <- g.indices) {
      for (j <- g.head.indices) {
        for (k <- g.head.head.indices) {
          g(i)(j)(k) = Random.nextDouble()
        }
      }
    }
    g
  }

  /**
   * Indexed vectors to BDM (the row number is known)
   * @param rowNum the row number of the matrix
   * @param indexedVectors the indexed vectors
   */
  def IndexedVec2BDM(rowNum: Int, indexedVectors: Array[(Int, Vector)]): BDM[Double] ={
    val colNum = indexedVectors.head._2.toArray.length
    val M: BDM[Double] = BDM.zeros(rowNum, colNum)
    IndexedVec2BDM(M, indexedVectors)
    M
  }

  /**
   * get other modes
   */
  def getOtherModes(mode: Int): (Int, Int) = {
    val i1: Int = (mode + 1) % 3
    val i2: Int = (mode + 2) % 3
    (i1, i2)
  }



  /**
   * transfer IndexedRowMatrix to Map[Long, BDV[Double]]
   */
  def IRMtoMap(irm: IndexedRowMatrix): Map[Long, BDV[Double]] = {
    val bdvMap = scala.collection.mutable.Map[Long, BDV[Double]]()

    val tmp: Array[IndexedRow] = irm.rows.collect()

    tmp.foreach { x =>
      bdvMap += (x.index -> Vector2BDV(x.vector))
    }

    bdvMap.toMap
  }

  /**
   * transfer Vector to BDV
   */
  def Vector2BDV(vector: Vector): BDV[Double] = {
    new BDV[Double](vector.toArray)
  }

  /**
   * initial FM in IndexedRowMatrix (index start from 0)
   */
  def initialFM(dim: Long, rank: Int)
               (implicit sc: SparkContext): IndexedRowMatrix = {

    val rowMatrix = initialRowMatrix(dim, rank)
    val map = rowMatrix.rows.zipWithIndex()
      .map { case (x, y) => IndexedRow(y, Vectors.dense(x.toArray)) }

    new IndexedRowMatrix(map)
  }

  /**
   * initial FM in IndexedRowMatrix  (index start from offset)
   */
  def initialFM(size: Long, rank: Int, offset: Int)
               (implicit sc: SparkContext): IndexedRowMatrix = {
    val rowMatrix = initialRowMatrix(size, rank)
    val map: RDD[IndexedRow] = rowMatrix.rows.zipWithIndex()
      .map { case (x, y) => IndexedRow(y + offset, Vectors.dense(x.toArray)) }

    new IndexedRowMatrix(map)
  }

  /**
   * initial a RowMatrix
   *
   * @param size dim of the mode
   * @param rank rank of the mode
   * @return
   */
  def initialRowMatrix(size: Long,
                       rank: Int)
                      (implicit sc: SparkContext): RowMatrix = {

    val rowData: RDD[Vector] = RandomRDDs.uniformVectorRDD(sc, size, rank)
      .map(x => Vectors.dense(BDV.rand[Double](rank).toArray))

    new RowMatrix(rowData, size, rank)
  }
}
