ThisBuild / useCoursier := false

name := "DITTD"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "com.github.scopt" %% "scopt" % "3.5.0",
  "net.sf.trove4j" % "core" % "3.1.0",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)