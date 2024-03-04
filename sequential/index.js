import tf from '@tensorflow/tfjs-node'
import fs from 'fs'
import path from 'path'
const __dirname = path.resolve()

let arquivo = fs.readFileSync(
  path.resolve(__dirname, 'sequential', 'bbse3-cotacao.csv'),
  {
    encoding: 'utf8',
  },
)

// Formato colunas do CSV
//DATA;ABERTURA;FECHAMENTO;MÍNIMO;MÁXIMO
arquivo = arquivo.toString().trim()

const linhas = arquivo.split('\n')
let X = []
let Y = []
let qtdLinhas = 0
for (let l = 1; l < linhas.length; l++) {
  let celulas1 = []
  if (qtdLinhas == linhas.length - 2) {
    // Deve ser recostada e inserida aqui a ultima data da lista do CSV
    // Separando por virgula
    celulas1 = ['18.12.2023', 30.6, 31.35, 30.55, 31.58]
  }
  // Casas decimais são duas nos dados obtidos no CSV
  else celulas1 = linhas[l + 1].split(';')
  const celulas2 = linhas[l].split(';')

  const AberturaX = Number(celulas1[1])
  const FechamentoX = Number(celulas1[2])
  const MinimaX = Number(celulas1[3])
  const MaximaX = Number(celulas1[4])
  X.push([FechamentoX, AberturaX, MaximaX, MinimaX])

  const AberturaY = Number(celulas2[1])
  const FechamentoY = Number(celulas2[2])
  const MinimaY = Number(celulas2[3])
  const MaximaY = Number(celulas2[4])
  Y.push([FechamentoY, AberturaY, MaximaY, MinimaY])

  qtdLinhas++
}

const model = tf.sequential()
const inputLayer = tf.layers.dense({ units: 4, inputShape: [4] })
model.add(inputLayer)
// Pós ponto colocar o dobro de ZEROS das casas decimais dos valores do CSV
// Pode aumentar para tentar um valor de maior precisão
const learningRate = 0.00001
const optimizer = tf.train.sgd(learningRate)
const compile = { loss: 'meanSquaredError', optimizer: optimizer }
model.compile(compile)
const x = tf.tensor(X, [qtdLinhas, 4])
const y = tf.tensor(Y)

// Dados para treino
// 01.03.2024;33.35;32.92;32.86;33.41   ==> Usado para ver a resposta se esta precisa
// 29.02.2024;33.54;33.34;33.31;33.68   ==> Usado para predição
const arrInput = [[33.54, 33.34, 33.31, 33.68]]
const input = tf.tensor(arrInput, [1, 4])

const result = [
  {
    epochs: -1,
    data: arrInput[0],
  },
]

// epochs: pode aumentar para tentar alcancar maior precisao
// porem chega até um número que mesmo se aumentar nao é efetivo
await model.fit(x, y, { epochs: 600 }).then(() =>
  result.push({
    epochs: 600,
    data: model
      .predict(input)
      .dataSync()
      .map((item) => Number(item)),
  }),
)

await model.fit(x, y, { epochs: 300 }).then(() =>
  result.push({
    epochs: 300,
    data: model
      .predict(input)
      .dataSync()
      .map((item) => Number(item)),
  }),
)

await model.fit(x, y, { epochs: 150 }).then(() =>
  result.push({
    epochs: 150,
    data: model
      .predict(input)
      .dataSync()
      .map((item) => Number(item)),
  }),
)

await model.fit(x, y, { epochs: 70 }).then(() =>
  result.push({
    epochs: 70,
    data: model
      .predict(input)
      .dataSync()
      .map((item) => Number(item)),
  }),
)

await model.fit(x, y, { epochs: 35 }).then(() =>
  result.push({
    epochs: 35,
    data: model
      .predict(input)
      .dataSync()
      .map((item) => Number(item)),
  }),
)

await model.fit(x, y, { epochs: 15 }).then(() =>
  result.push({
    epochs: 15,
    data: model
      .predict(input)
      .dataSync()
      .map((item) => Number(item)),
  }),
)

let resume = result
  .map((item) => `epochs: ${item.epochs} ; \t ${item.data.join('\t\t')}`)
  .join('\n')

console.log(resume)
