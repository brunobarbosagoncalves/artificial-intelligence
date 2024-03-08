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
    this.learningRate = params?.learningRate || 0.00001
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

  async Start() {
    // Load file
    this.LoadFile()

    // Get header from line first line
    this.HeaderFromLines()

    // Get index of columns with numbers
    this.ColumnsWithNumbers()

    // Remove coluns no number
    this.CleanDataAndHeaders()

    // Remove coluns no number
    this.ConvertLinesToNumber()

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
    const compile = { loss: 'meanSquaredError', optimizer }
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

    return {
      headers: this.headers,
      lineTarget,
      linePredict,
      lineAfterTarget,
    }
  }
}

const __dirname = path.resolve()

const pathCsv = path.resolve(__dirname, 'sequential', 'bbse3-cotacao.csv')

const results = await Promise.all([
  new Sequential({ pathCsv, epochs: 250 }).Start(),
  new Sequential({ pathCsv, epochs: 500 }).Start(),
  // new Sequential({ pathCsv, epochs: 1000 }).Start(),
  // new Sequential({ pathCsv, epochs: 2000 }).Start(),
  // new Sequential({ pathCsv, epochs: 4000 }).Start(),
  // new Sequential({ pathCsv, epochs: 8000 }).Start(),
  // new Sequential({ pathCsv, epochs: 16000 }).Start(),
])

console.log(JSON.stringify(results, null, 2))
