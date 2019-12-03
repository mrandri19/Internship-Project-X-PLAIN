import React, { useState, useEffect } from "react"
import "whatwg-fetch"
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link,
  Redirect
} from "react-router-dom"
import Plot from "react-plotly.js"
import Button from "@material-ui/core/Button"
import Box from "@material-ui/core/Box"
import ThemeProvider from "@material-ui/styles/ThemeProvider"
import { createMuiTheme } from "@material-ui/core/styles"
import AppBar from "@material-ui/core/AppBar"
import Tabs from "@material-ui/core/Tabs"
import "./App.css"

import Toolbar from "@material-ui/core/Toolbar"
import Tab from "@material-ui/core/Tab"
import Typography from "@material-ui/core/Typography"
import List from "@material-ui/core/List"
import ListItem from "@material-ui/core/ListItem"
import ListItemText from "@material-ui/core/ListItemText"

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
    <Box m={1}>
      <Typography variant="h6">Datasets</Typography>
      <List component="ul">
        {datasets.map(datasetName => (
          <ListItem button key={datasetName} onClick={postDataset(datasetName)}>
            <ListItemText primary={datasetName} />
          </ListItem>
        ))}
      </List>
    </Box>
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

const theme = createMuiTheme({})

function LinkTab(props) {
  return <Tab component={Link} {...props} />
}

function App() {
  return (
    <Router>
      <ThemeProvider theme={theme}>
        <Route path="/">
          <>
            <AppBar position="sticky">
              <Toolbar>
                <Typography variant="h6">LACE</Typography>
                <Tabs value={window.location.pathname}>
                  <LinkTab label="Datasets" to="/datasets" />
                  <LinkTab label="Classifiers" to="/classifiers" />
                  <LinkTab label="Instances" to="/instances" />
                  <LinkTab label="Explanation" to="/explanation" />
                </Tabs>
              </Toolbar>
            </AppBar>

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
          </>
        </Route>
      </ThemeProvider>
    </Router>
  )
}

export default App
