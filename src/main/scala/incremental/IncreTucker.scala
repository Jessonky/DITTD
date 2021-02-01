package incremental

import java.util.Date

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import utils.Utils._

/**
 * Parse the params, and run the incremental tucker decomposition.
 */
object IncreTucker {

  private case class Params(Mode: Int = 3,
                            ranks: Array[Int] = Array[Int](3, 3, 3),
                            cutNum: Int = 12,
                            partAlg: Int = 0,
                            mu: Double = 0.8,
                            inputPre: String = "",
                            inputIncre: String = "",
                            output: String = "")

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("IncreTucker") {
      head("Incremental Tucker decomposition on Spark")

      // set the mode of tensor
      opt[Int]('m', "mode")
        .text(s"Tensor's mode. default: ${defaultParams.Mode}")
        .action((x, c) => c.copy(Mode = x))
        .validate(x =>
          if (x > 0) success
          else failure("number of mode m must be positive."))

      // set the ranks
      opt[String]('r', "ranks")
        .text(
          s"The ranks of Tucker Decomposition. default: ${defaultParams.ranks}")
        .text("shape")
        .action((x, c) => c.copy(ranks = x.split(',').map(_.toInt)))

      // set the number of partitions
      opt[Int]('c', "cutNum")
        .text(s"The cutNum of the tensor. default: ${defaultParams.cutNum}")
        .action((x, c) => c.copy(cutNum = x))
        .validate(x =>
          if (x > 0) success
          else failure("number of cutNum c must be positive."))

      // set the partitioning method
      opt[Int]('a', "partAlg")
        .text(s"The partitioning algorithm for the tensor. default: ${defaultParams.partAlg}, bucket" +
          s" partitioning")
        .action((x, c) => c.copy(partAlg = x))
        .validate(x =>
          if (x.equals(0) || x.equals(1)) success
          else failure("partitioning algorithm a must be 0 or 1"))

      // set the \alpha
      opt[Double]("mu")
        .text(s"Forgetting coefficient for the pre-tensor. default: ${defaultParams.mu}")
        .action((x, c) => c.copy(mu = x))
        .validate(x =>
          if (x > 0.0) success
          else failure("mu must be positive."))

      // sent the input path of pre tensor
      opt[String]('p', "inputPre")
        .required()
        .text("path of pre-input file.")
        .action((x, c) => c.copy(inputPre = x))

      // sent the input path of incremental tensor
      opt[String]('i', "inputIncre")
        .required()
        .text("path of incre-input file.")
        .action((x, c) => c.copy(inputIncre = x))

      // sent the output path
      opt[String]('o', "output")
        .required()
        .text(s"output write path.")
        .action((x, c) => c.copy(output = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => increTucker(params)
      case None =>
        parser.showUsageAsError()
        sys.exit(1)
    }
  }


  private def increTucker(params: Params): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("localIncreCP").setMaster("local[*]") //local
//    val conf = new SparkConf().set("spark.driver.maxResultSize", "4g") //call by spark-submit
//    val conf = new SparkConf() //call by spark-submit
    implicit val sc: SparkContext = new SparkContext(conf)
    sc.setLogLevel("WARN")


    println("//////////////////////// The parameter settings ////////////////////////////////")
    println("input pre tensor path: " +  params.inputPre)
    println("input incremental tensor path: " +  params.inputIncre)
    println("the mode: " +  params.Mode)
    println("the ranks: " +  params.ranks.toList)
    println("the partition number: " +  params.cutNum)
    println("the partition algorithm (0 for bucket, 1 for greedy): " +  params.partAlg)
    println("the mu: " +  params.mu)
    println("output path:  " +  params.output)
    println("////////////////////////////////////////////////////////////////////////////////")

    val preTensor = fromFile(params.inputPre)
    val preDims: Array[Int] = getDims(preTensor, params.Mode)
    val FM0s = new Array[BDM[Double]](params.Mode)
    for(i <- FM0s.indices){
      FM0s(i) = BDM.rand(preDims(i), params.ranks(i))
      modifiedGramSchmidt(FM0s(i))
    }
    val g = initialCoordinatesG(params.ranks)

    val curTensor = fromFile(params.inputIncre) // incremental tensor

    val startIncre = new Date().getTime
    val (g1, fm1s, partitionTime, partitionCV) = new IncreTuckerImpl()
      .setMode(params.Mode)
      .setRanks(params.ranks)
      .setMu(params.mu)
      .setCutNum(params.cutNum)
      .setPartAlg(params.partAlg)
      .run(FM0s, g, preDims, curTensor)
    val endIncre = new Date().getTime
    println("---------------------------------------------------------------------------------")
    println("Incremental Tucker Decomposition running time: " +
      ((endIncre - startIncre).toDouble / 1000.0).formatted("%.2f"))
    println("partition time: " + (partitionTime.toDouble / 1000.0).formatted("%.2f"))
    println("partition CV: " + partitionCV.formatted("%.8f"))
  }
}
