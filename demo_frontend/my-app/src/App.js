import React, { useState, useEffect } from "react"
import "whatwg-fetch"
import {
  BrowserRouter as Router,
  Switch,
  Route,
  NavLink,
  Redirect
} from "react-router-dom"
import Plot from "react-plotly.js"

function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [toClassifiers, setToClassfiers] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/datasets")
      const json = await res.json()
      setDatasets(json)
    }
    fetchData()
  }, [])

  function postDataset(datasetName) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/dataset/${datasetName}`, {
        method: "POST"
      })
      setToClassfiers(true)
    }
  }

  if (toClassifiers) {
    return <Redirect to="/classifiers" />
  }
  return (
    <>
      <h2>Datasets</h2>
      <ul>
        {datasets.map(datasetName => (
          <li key={datasetName}>
            <button onClick={postDataset(datasetName)}>{datasetName}</button>
          </li>
        ))}
      </ul>
    </>
  )
}

function Classifiers() {
  const [classifiers, setClassifiers] = useState([])
  const [toInstances, setToInstances] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/classifiers")
      const json = await res.json()
      setClassifiers(json)
    }
    fetchData()
  }, [])

  function postClassifier(datasetName) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/classifier/${datasetName}`, {
        method: "POST"
      })
      setToInstances(true)
    }
  }

  if (toInstances) {
    return <Redirect to="/instances" />
  }
  return (
    <>
      <h2>Classifiers</h2>
      <ul>
        {classifiers.map(classifierName => (
          <li key={classifierName}>
            <button onClick={postClassifier(classifierName)}>
              {classifierName}
            </button>
          </li>
        ))}
      </ul>
    </>
  )
}

function Instances() {
  const [instances, setInstances] = useState([])
  const [toExplanation, setToExplanation] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/instances")
      const json = await res.json()
      setInstances(json)
    }
    fetchData()
  }, [])

  function postInstance(instanceId) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/instance/${instanceId}`, {
        method: "POST"
      })
      setToExplanation(true)
    }
  }

  if (toExplanation) {
    return <Redirect to="/explanation" />
  }
  return (
    <>
      <h2>Instances</h2>
      <table>
        <thead></thead>
        <tbody>
          {instances.slice(0, 20).map(instance => (
            <tr key={instance[1]}>
              {instance[0].map((feature, feature_ix) => (
                <td key={instance[1] + feature_ix}>{feature}</td>
              ))}
              <td>
                <button onClick={postInstance(instance[1])}>Pick</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function Explanation() {
  const [explanation, setExplanation] = useState(null)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/explanation")
      const json = await res.json()
      console.log(json)
      setExplanation(json)
    }
    fetchData()
  }, [])

  if (explanation === null) {
    return (
      <>
        <h2>Explanation</h2>
        <p>Loading</p>
      </>
    )
  }

  const differences = explanation.diff_single.concat(
    Object.values(explanation.map_difference)
  )

  const names = Array.from(Array(explanation.diff_single.length).keys())
    .map(i => `${i + 1}`)
    .concat(Object.keys(explanation.map_difference).map((val, ix) => `${val}`))

  const trace = {
    type: "bar",
    x: differences,
    y: names,
    orientation: "h",
    marker: {
      color: names.map(d => (d.includes(",") ? "red" : "blue"))
    }
  }

  console.log(names)
  return (
    <>
      <h2>Explanation</h2>
      <h3>Classifier Prediction</h3>
      <p>
        The instance {explanation.instance_id} belongs to the class{" "}
        {explanation.target_class} with probability{" "}
        {explanation.prob.toFixed(3)}.
      </p>
      <h3>LACE Explanation</h3>
      <p>
        The method has converged with error {explanation.error.toFixed(3)} and a
        locality size (parameter K) of {explanation.k}.
      </p>
      <p></p>
      <Plot
        data={[trace]}
        layout={{ yaxis: { type: "category", automargin: true } }}
      />
    </>
  )
}

function App() {
  return (
    <Router>
      <div>
        <h1>LACE</h1>

        <nav>
          <ul>
            <li>
              <NavLink to="/datasets" activeClassName="nav-active">
                Datasets
              </NavLink>
            </li>
            <li>
              <NavLink to="/classifiers" activeClassName="nav-active">
                Classifiers
              </NavLink>
            </li>
            <li>
              <NavLink to="/instances" activeClassName="nav-active">
                Instances
              </NavLink>
            </li>
            <li>
              <NavLink to="/explanation" activeClassName="nav-active">
                Explanation
              </NavLink>
            </li>
          </ul>
        </nav>

        <Switch>
          <Route path="/datasets">
            <Datasets />
          </Route>

          <Route path="/classifiers">
            <Classifiers />
          </Route>

          <Route path="/instances">
            <Instances />
          </Route>

          <Route path="/explanation">
            <Explanation />
          </Route>

          <Route path="/">
            <Redirect to="/datasets" />
          </Route>
        </Switch>
      </div>
    </Router>
  )
}

export default App
