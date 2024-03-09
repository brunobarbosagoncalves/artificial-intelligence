import tf from '@tensorflow/tfjs-node'
import fs from 'fs'
import path from 'path'

class Sequential {
  /**
   *
   *  @param {Object} params
   *  @param {String} params.pathCsv /fullpath/to/myfile.csv
   *  @param {Object} params.configCsv { encoding: 'utf8' }
   *  @param {Number} params.epochs 100
   *  @param {Float[Float]} params.learningRate 0.00001
   */

  constructor(params) {
    this.pathCsv = params?.pathCsv
    this.configCsv = params?.configCsv || { encoding: 'utf8' }
    this.epochs = params?.epochs || 500
    this.learningRate = params?.learningRate || false
    this.columnsWithNumbers = []
    this.headers = []
    this.lines = []
  }

  LoadFile() {
    const file = fs.readFileSync(this.pathCsv, this.configCsv).toString().trim()

    this.lines = file.split('\n')
  }

  HeaderFromLines() {
    this.headers = this.lines.shift().split(';')
  }

  ColumnsWithNumbers() {
    const positions = this.lines[0]
      .split(';')
      .reduce((a, c, i) => (!!Number(c) ? [...a, i] : [...a]), [])

    this.columnsWithNumbers = positions
  }

  CleanDataAndHeaders() {
    const positions = this.columnsWithNumbers
    this.headers = this.headers.filter((h, i) => positions.includes(i))
    this.lines = this.lines.map((line, i) =>
      line
        .split(';')
        .filter((_, i) => positions.includes(i))
        .join(';'),
    )
  }

  ConvertLinesToNumber() {
    this.lines = this.lines.map((line) => this.ConvertLineToNumberList(line))
  }

  ConvertLineToNumberList(line) {
    return line.split(';').map((n) => Number(n))
  }

  VerifyBadNumbers() {
    const lineError = this.lines.find(
      (lineList) => lineList.find((item) => !Number(item)) !== undefined,
    )

    if (lineError) throw `Linha contem valores não numéricos: ${lineError}`
  }

  CalculeLearningRate() {
    // If has defined on instance
    if (this.learningRate) return true

    // Calcule
    const linesToCalculeLearningRate = this.lines.slice(0, 3)
    const greaterDecimal = linesToCalculeLearningRate.reduce((a, c, i) => {
      c.forEach((number) => {
        let totalDecimal = number.toString().split('.')[1].length || 1
        if (totalDecimal > a) a = totalDecimal
      })
      return a
    }, 1)

    this.learningRate = Number(
      '0.'.padEnd(greaterDecimal * 2 + 2, '0').concat('1'),
    )
  }

  async Start() {
    // Load file
    this.LoadFile()

    // Get header from line first line
    this.HeaderFromLines()

    // Get index of columns with numbers
    this.ColumnsWithNumbers()

    // Remove coluns not number
    this.CleanDataAndHeaders()

    // Remove coluns no number
    this.ConvertLinesToNumber()

    // Verify bad number on lines
    this.VerifyBadNumbers()

    // Calcule LearningRate using decimal length
    this.CalculeLearningRate()

    // Last line format array
    const lineLast = this.lines.pop()

    // target to predict
    const lineTarget = this.lines.shift()

    // line before target
    const lineAfterTarget = this.lines.shift()

    const lengthNumbersData = lineAfterTarget.length

    let X = []
    let Y = []
    let countLines = 0
    for (let l = 1; l < this.lines.length; l++) {
      let line1 = []
      let line2 = []

      // Deve ser recortada e inserida aqui a ultima data da lista do CSV
      // Separando por virgula
      if (countLines == this.lines.length - 2) line1 = lineLast
      // Casas decimais são duas nos dados obtidos no CSV
      else line1 = this.lines[l + 1]

      line2 = this.lines[l]

      X.push(line1)
      Y.push(line2)

      countLines++
    }

    const model = tf.sequential()
    const inputLayer = tf.layers.dense({
      units: lengthNumbersData,
      inputShape: [lengthNumbersData],
    })
    model.add(inputLayer)
    // Pós ponto colocar o dobro de ZEROS das casas decimais dos valores do CSV
    // Pode aumentar para tentar um valor de maior precisão
    const optimizer = tf.train.sgd(this.learningRate)
    const compile = {
      loss: 'meanSquaredError',
      optimizer,
      metrics: ['accuracy'],
    }
    model.compile(compile)
    const x = tf.tensor(X, [countLines, lengthNumbersData])
    const y = tf.tensor(Y)

    // Dados para treino
    // 01.03.2024;33.35;32.92;32.86;33.41   ==> Usado para ver a resposta se esta precisa
    // 29.02.2024;33.54;33.34;33.31;33.68   ==> Usado para predição
    const arrInput = [lineTarget]
    const input = tf.tensor(arrInput, [1, lengthNumbersData])

    // epochs: pode aumentar para tentar alcancar maior precisao
    // porem chega até um número que mesmo se aumentar nao é efetivo
    await model.fit(x, y, { epochs: this.epochs })

    const linePredictRaw = model
      .predict(input)
      .dataSync()
      .reduce((a, c) => [...a, parseFloat(c).toFixed(2)], [])

    const linePredict = this.ConvertLineToNumberList(linePredictRaw.join(';'))

    const accuracy = lineTarget.map((lt, i) =>
      Number((lt - linePredict[i]).toFixed(2)),
    )

    const accuracyTotal = Number(accuracy.reduce((a, c) => a + c, 0)).toFixed(2)

    return {
      headers: this.headers,
      lineTarget,
      linePredict,
      lineAfterTarget,
      accuracy,
      accuracyTotal,
      epochs: this.epochs,
    }
  }
}
// Format tabulation on array
const ListToTabulation = (list) =>
  list.map((h) => h.toString().padEnd(10, ' ')).join('\t')

const TransformResultToString = (result) => {
  const data = []
  const headers = [...result.headers, 'INFOS']

  data.push(ListToTabulation(headers.map((i) => i.slice(0.6))))
  data.push(ListToTabulation(result.lineTarget))
  data.push(ListToTabulation(headers.map(() => '- - - - - - - -')))

  result.data.forEach((d) => {
    data.push(
      ListToTabulation([
        ...d.linePredict,
        `[accur.: ${d.accuracyTotal} ; epochs: ${d.epochs}]`,
      ]),
    )
  })

  return data.join('\n')
}

const FormatResult = (resultList) =>
  resultList.reduce(
    (a, c, i) => ({
      headers: c.headers,
      lineTarget: c.lineTarget,
      lineAfterTarget: c.lineAfterTarget,
      data: [
        ...a.data,
        {
          epochs: c.epochs,
          accuracyTotal: c.accuracyTotal,
          accuracy: c.accuracy,
          linePredict: c.linePredict,
        },
      ],
    }),
    { data: [], headers: false, lineTarget: false, lineAfterTarget: false },
  )

const __dirname = path.resolve()

const pathCsv = path.resolve(__dirname, 'sequential', 'cotacao.csv')

/**
 * O processo é executado usando dois parametros base
 * pathCsv: csv contendo colunas numericas a ser processadas
 * epochs: o quanto o algoritmo vai estudar a lógica dos dados numericos
 *
 * epochs
 *   - Quando a epochs é muito baixo o algoritmo não consegue atingir uma acuracia boa
 *   - Quando a epochs é muito alta o algoritmo fica improdutivo e disperdiça processamento em acuracia defasada
 *
 * Cuidado:
 *   - A maioria das linhas com epochs altas foram comentadas para nao ser executada por acidente pois
 *     consome muito processo e memoria
 */
const results = await Promise.all([
  new Sequential({ pathCsv, epochs: 50 }).Start(),
  new Sequential({ pathCsv, epochs: 100 }).Start(),
  // new Sequential({ pathCsv, epochs: 250 }).Start(),
  // new Sequential({ pathCsv, epochs: 500 }).Start(),
  // new Sequential({ pathCsv, epochs: 1000 }).Start(),
  // new Sequential({ pathCsv, epochs: 2000 }).Start(),
  // new Sequential({ pathCsv, epochs: 4000 }).Start(),
  // new Sequential({ pathCsv, epochs: 8000 }).Start(),
  // new Sequential({ pathCsv, epochs: 16000 }).Start(),
  // new Sequential({ pathCsv, epochs: 32000 }).Start(),
])
  // Format all line in one object
  .then((param) => FormatResult(param))
  // Format result to string
  .then((param) => TransformResultToString(param))

console.log(results)
