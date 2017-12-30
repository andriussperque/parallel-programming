import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object Analisador {

  // Args = path/to/text0.txt path/to/text1.txt
  def main(args: Array[String]) {

    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Contagem de Palavra"))

    /**
        A implementação deste código foi realizada em passos, para uma prosta educativa.
        O nome de cada variável resume o que o trecho de código realizada naquele passo.
    */


    // Parte 1: contar palavras do texto 1 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE) 
    // imprimir na cada linha: "palavra=numero"

    println("TEXT1")
 
    val lines1 = sc.textFile(args(0))                                       // Lê o arquivo

    val words = lines1.flatMap(line => line.split(" "))                     // Quebra o texto de cada linha em palavras.

    val wordsFiltered1 = words.map(word => word.replaceAll("[,.!?:;]","").toLowerCase)      // Filtra caracteres indesejados e deixa em Str em minúsculo

    val result1 = wordsFiltered1.map(word => (word,1)).reduceByKey(_+_)     // Realiza a contagem das palavras semelhantes

    val result1bigger3 = result1.filter { case (x, _) => x.length > 3 }     // Filtra por palavras com tamanho maior do que 3 letras

    val result1Sorted = result1bigger3.sortBy(- _._2)                       // Ordena contador por ordem decrescente

    val result1Cutted = result1Sorted.take(5)                               // Seleciona as 5 primeiras entradas

    result1Cutted.foreach{ case (key, count) => println(key+"="+count)}     // Imprime no formato desejado

    // Parte 2: contar palavras do texto 2 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE) 
    // imprimir na cada linha: "palavra=numero"

    println("TEXT2")

    val lines2 = sc.textFile(args(1))

    val words2 = lines2.flatMap(line => line.split(" "))

    val wordsFiltered2 = words2.map(word => word.replaceAll("[,.!?:;]","").toLowerCase)

    val result2 = wordsFiltered2.map(word => (word,1)).reduceByKey(_+_)

    val result2bigger3 = result2.filter { case (x, _) => x.length > 3 } 

    val result2Sorted = result2bigger3.sortBy(- _._2)

    val result2Cutted = result2Sorted.take(5)

    result2Cutted.foreach{ case (key, count) => println(key+"="+count)}

    // Part 3: comparar resultado e imprimir na ordem ALFABETICA todas as palavras que aparecem MAIS que 100 vezes nos 2 textos imprimir na cada linha: "palavra"
    println("COMMON")

    val wordsBigger1 = result1bigger3.filter { case (_, y) => y > 100 } .keys

    val wordsBigger2 = result2bigger3.filter { case (_, y) => y > 100 } .keys

    val wordsIntersection = wordsBigger1.intersection(wordsBigger2)         // Realiza a intersecção das palavras com contador maior do que 100 de cada texto

    val wordsSorted = wordsIntersection.collect().toList.sorted             // Ordena em ordem crescente a lista de palavras.

    wordsSorted.foreach(println)                                            // Realiza a impressão final.

  }
}

/**
alice30 warw10
TEXT1
said=456
alice=377
that=234
with=172
very=139
TEXT2
that=759
with=448
were=365
from=326
they=302
COMMON
little
said
that
they
this
with

alice30 wizoz10
TEXT1
said=456
alice=377
that=234
with=172
very=139
TEXT2
they=390
that=350
dorothy=340
said=331
with=274
COMMON
little
said
that
they
this
with

warw10  wizoz10
TEXT1
that=759
with=448
were=365
from=326
they=302
TEXT2
they=390
that=350
dorothy=340
said=331
with=274
COMMON
came
could
from
have
little
said
that
them
then
there
they
this
were
with

alice30 gmars11
TEXT1
said=456
alice=377
that=234
with=172
very=139
TEXT2
that=1235
with=633
from=599
upon=523
were=458
COMMON
said
that
they
this
with

warw10  gmars11
TEXT1
that=759
with=448
were=365
from=326
they=302
TEXT2
that=1235
with=633
from=599
upon=523
were=458
COMMON
about
again
been
black
came
could
from
have
into
said
that
their
them
then
there
they
this
through
time
upon
were
with

wizoz10 gmars11
TEXT1
they=390
that=350
dorothy=340
said=331
with=274
TEXT2
that=1235
with=633
from=599
upon=523
were=458
COMMON
came
could
from
great
have
said
that
them
then
there
they
this
were
when
will
with
would

*/






