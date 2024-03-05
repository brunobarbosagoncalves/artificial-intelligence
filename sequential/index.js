// import tf from '@tensorflow/tfjs-node'
// import fs from 'fs'
// import path from 'path'

// const __dirname = path.resolve()
// const pathCsv = [__dirname, 'sequential', 'bbse3-cotacao.csv']
// const configCsv = { encoding: 'utf8' }
// let arquivo = fs.readFileSync(path.resolve(pathCsv), configCsv)

// // Formato colunas do CSV
// //DATA;ABERTURA;FECHAMENTO;MÍNIMO;MÁXIMO
// arquivo = arquivo.toString().trim()

// const linhas = arquivo.split('\n')
// let X = []
// let Y = []
// let qtdLinhas = 0
// for (let l = 1; l < linhas.length; l++) {
//   let celulas1 = []
//   // Deve ser recostada e inserida aqui a ultima data da lista do CSV
//   // Separando por virgula
//   if (qtdLinhas == linhas.length - 2)
//     celulas1 = ['18.12.2023', 30.6, 31.35, 30.55, 31.58]
//   // Casas decimais são duas nos dados obtidos no CSV
//   else celulas1 = linhas[l + 1].split(';')

//   const celulas2 = linhas[l].split(';')

//   const AberturaX = Number(celulas1[1])
//   const FechamentoX = Number(celulas1[2])
//   const MinimaX = Number(celulas1[3])
//   const MaximaX = Number(celulas1[4])
//   X.push([FechamentoX, AberturaX, MaximaX, MinimaX])

//   const AberturaY = Number(celulas2[1])
//   const FechamentoY = Number(celulas2[2])
//   const MinimaY = Number(celulas2[3])
//   const MaximaY = Number(celulas2[4])
//   Y.push([FechamentoY, AberturaY, MaximaY, MinimaY])

//   qtdLinhas++
// }

// const model = tf.sequential()
// const inputLayer = tf.layers.dense({ units: 4, inputShape: [4] })
// model.add(inputLayer)
// // Pós ponto colocar o dobro de ZEROS das casas decimais dos valores do CSV
// // Pode aumentar para tentar um valor de maior precisão
// const learningRate = 0.00001
// const optimizer = tf.train.sgd(learningRate)
// const compile = { loss: 'meanSquaredError', optimizer: optimizer }
// model.compile(compile)
// const x = tf.tensor(X, [qtdLinhas, 4])
// const y = tf.tensor(Y)

// // Dados para treino
// // 01.03.2024;33.35;32.92;32.86;33.41   ==> Usado para ver a resposta se esta precisa
// // 29.02.2024;33.54;33.34;33.31;33.68   ==> Usado para predição
// const arrInput = [[33.54, 33.34, 33.31, 33.68]]
// const input = tf.tensor(arrInput, [1, 4])

// const result = [
//   {
//     epochs: -1,
//     data: arrInput[0],
//   },
// ]

// // epochs: pode aumentar para tentar alcancar maior precisao
// // porem chega até um número que mesmo se aumentar nao é efetivo
// await model.fit(x, y, { epochs: 600 }).then(() =>
//   result.push({
//     epochs: 600,
//     data: model
//       .predict(input)
//       .dataSync()
//       .map((item) => Number(item)),
//   }),
// )

// await model.fit(x, y, { epochs: 300 }).then(() =>
//   result.push({
//     epochs: 300,
//     data: model
//       .predict(input)
//       .dataSync()
//       .map((item) => Number(item)),
//   }),
// )

// await model.fit(x, y, { epochs: 150 }).then(() =>
//   result.push({
//     epochs: 150,
//     data: model
//       .predict(input)
//       .dataSync()
//       .map((item) => Number(item)),
//   }),
// )

// await model.fit(x, y, { epochs: 70 }).then(() =>
//   result.push({
//     epochs: 70,
//     data: model
//       .predict(input)
//       .dataSync()
//       .map((item) => Number(item)),
//   }),
// )

// await model.fit(x, y, { epochs: 35 }).then(() =>
//   result.push({
//     epochs: 35,
//     data: model
//       .predict(input)
//       .dataSync()
//       .map((item) => Number(item)),
//   }),
// )

// await model.fit(x, y, { epochs: 15 }).then(() =>
//   result.push({
//     epochs: 15,
//     data: model
//       .predict(input)
//       .dataSync()
//       .map((item) => Number(item)),
//   }),
// )

// let resume = result
//   .map((item) => `epochs: ${item.epochs} ; \t ${item.data.join('\t\t')}`)
//   .join('\n')

// console.log(resume)

//----------------------

import tf from '@tensorflow/tfjs-node'
import fs from 'fs'
import path from 'path'

class Sequential {
  /**
   *
   *  @param {Object} params
   *  @param {String} params.pathCsv /fullpath/to/myfile.csv
   *  @param {Object} params.configCsv { encoding: 'utf8' }
   *  @param {Array[Object]} params.mapCsv [{ name:'ColumnName', position: ColumnPosition, ignore: trueOrFalse  }]
   *  @param {Number} params.epochs 100
   *  @param {Float[Float]} params.learningRate 0.00001
   */

  constructor(params) {
    this.pathCsv = params?.pathCsv
    this.configCsv = params?.configCsv
    this.mapPositionsCsv = params?.mapCsv
      .filter((i) => !i.ignore)
      .map((i) => i.position)
    this.mapColumnsCsv = params?.mapCsv.map((i) => i.column)
    this.epochs = params?.epochs
    this.learningRate = params?.learningRate
  }

  filterPosition(data) {
    return data.reduce(
      (acc, cur, ind) =>
        this.mapPositionsCsv.includes(ind) ? [...acc, Number(cur)] : [...acc],
      [],
    )
  }

  async start() {
    // Load file
    let arquivo = fs.readFileSync(this.pathCsv, this.configCsv)

    // Formato colunas do CSV
    //DATA;ABERTURA;FECHAMENTO;MÍNIMO;MÁXIMO
    arquivo = arquivo.toString().trim()

    const linhas = arquivo.split('\n')

    // ultima linha formatada em array
    const linhaFinal = linhas.pop().split(';')
    const linhaAlvoPredicao = this.filterPosition(linhas.shift().split(';'))
    const linhaPreAlvoPredicao = this.filterPosition(linhas.shift().split(';'))

    let X = []
    let Y = []
    let qtdLinhas = 0
    for (let l = 1; l < linhas.length; l++) {
      let celulas1 = []
      let celulas2 = []

      // Deve ser recortada e inserida aqui a ultima data da lista do CSV
      // Separando por virgula
      if (qtdLinhas == linhas.length - 2) celulas1 = linhaFinal
      // Casas decimais são duas nos dados obtidos no CSV
      else celulas1 = linhas[l + 1].split(';')

      celulas2 = linhas[l].split(';')

      X.push(this.filterPosition(celulas1))
      Y.push(this.filterPosition(celulas2))

      qtdLinhas++
    }

    const model = tf.sequential()
    const inputLayer = tf.layers.dense({ units: 4, inputShape: [4] })
    model.add(inputLayer)
    // Pós ponto colocar o dobro de ZEROS das casas decimais dos valores do CSV
    // Pode aumentar para tentar um valor de maior precisão
    const optimizer = tf.train.sgd(this.learningRate)
    const compile = { loss: 'meanSquaredError', optimizer: optimizer }
    model.compile(compile)
    const x = tf.tensor(X, [qtdLinhas, 4])
    const y = tf.tensor(Y)

    // Dados para treino
    // 01.03.2024;33.35;32.92;32.86;33.41   ==> Usado para ver a resposta se esta precisa
    // 29.02.2024;33.54;33.34;33.31;33.68   ==> Usado para predição
    const arrInput = [linhaPreAlvoPredicao]
    const input = tf.tensor(arrInput, [1, 4])

    // epochs: pode aumentar para tentar alcancar maior precisao
    // porem chega até um número que mesmo se aumentar nao é efetivo
    await model.fit(x, y, { epochs: this.epochs })

    const linhaPredicao = model.predict(input).dataSync()

    console.log('TARGET::', linhaPreAlvoPredicao)
    console.log(
      'FINDED::',
      linhaPredicao.reduce(
        (acc, cur) => [...acc, Number(Number(cur).toFixed(2))],
        [],
      ),
    )
  }
}

const __dirname = path.resolve()

const seq = new Sequential({
  pathCsv: path.resolve(__dirname, 'sequential', 'bbse3-cotacao.csv'),
  configCsv: { encoding: 'utf8' },
  epochs: 500,
  learningRate: 0.00001,
  mapCsv: [
    { name: 'DATA', position: 0, ignore: true },
    { name: 'ABERTURA', position: 1 },
    { name: 'FECHAMENTO', position: 2 },
    { name: 'MÍNIMO', position: 3 },
    { name: 'MÁXIMO', position: 4 },
  ],
})

seq.start()
