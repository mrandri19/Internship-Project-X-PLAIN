import React, { useState, useEffect } from "react"
import "bootstrap/dist/css/bootstrap.min.css"
import "whatwg-fetch"
import { Switch, Route, Link, Redirect, useLocation } from "react-router-dom"
import Plot from "react-plotly.js"
import Navbar from "react-bootstrap/Navbar"
import Nav from "react-bootstrap/Nav"
import NavItem from "react-bootstrap/NavItem"
import Container from "react-bootstrap/Container"
import Button from "react-bootstrap/Button"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import ListGroup from "react-bootstrap/ListGroup"
import Spinner from "react-bootstrap/Spinner"
import Table from "react-bootstrap/Table"

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
    <Container>
      <Row>
        <Col className="mt-3">
          <h2>Datasets</h2>
          <ListGroup>
            {datasets.map(datasetName => (
              <ListGroup.Item
                action
                key={datasetName}
                onClick={postDataset(datasetName)}
              >
                {datasetName}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
    </Container>
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
    <Container>
      <Row>
        <Col className="mt-3">
          <h2>Classifiers</h2>
          <ListGroup>
            {classifiers.map(classifier => (
              <ListGroup.Item
                action
                key={classifier}
                onClick={postClassifier(classifier)}
              >
                {classifier}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
    </Container>
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

  if (instances.length === 0) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Instances</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  if (toExplanation) {
    return <Redirect to="/explanation" />
  }
  return (
    <Container>
      <Row>
        <Col className="mt-3">
          <h2>Instances</h2>
          <Table hover>
            <thead></thead>
            <tbody>
              {instances.slice(0, 20).map(instance => (
                <tr key={instance[1]}>
                  {instance[0].map((feature, feature_ix) => (
                    <td key={instance[1] + feature_ix}>{feature}</td>
                  ))}
                  <td>
                    <Button size="sm" onClick={postInstance(instance[1])}>
                      Get Explanation
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Col>
      </Row>
    </Container>
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
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Explanation</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = explanation.diff_single.concat(
    Object.values(explanation.map_difference)
  )

  const names = Array.from(Array(explanation.diff_single.length).keys())
    .map(i => `${i + 1}`)
    .concat(Object.keys(explanation.map_difference).map((val, _ix) => `${val}`))

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
    <Container>
      <Row>
        <Col className="mt-3">
          <h2>Explanation</h2>
          <h3>Classifier Prediction</h3>
          <p>
            The instance {explanation.instance_id} belongs to the class{" "}
            {explanation.target_class} with probability{" "}
            {explanation.prob.toFixed(3)}.
          </p>
          <h3>LACE Explanation</h3>
          <p>
            The method has converged with error {explanation.error.toFixed(3)}{" "}
            and a locality size (parameter K) of {explanation.k}.
          </p>
          <p></p>
          <Plot
            data={[trace]}
            layout={{ yaxis: { type: "category", automargin: true } }}
          />
        </Col>
      </Row>
    </Container>
  )
}

function App() {
  const location = useLocation()

  return (
    <Route path="/">
      <main>
        <Navbar bg="dark" variant="dark" expland="lg">
          <Navbar.Brand as={Link} to="/">
            LACE
          </Navbar.Brand>
          <Navbar.Collapse>
            <Nav activeKey={location.pathname} navbar={true}>
              <NavItem href="/datasets">
                <Nav.Link as={Link} eventKey="/datasets" to="/datasets">
                  Datasets
                </Nav.Link>
              </NavItem>
              <NavItem href="/classifiers">
                <Nav.Link as={Link} eventKey="/classifiers" to="/classifiers">
                  Classifiers
                </Nav.Link>
              </NavItem>
              <NavItem href="/instances">
                <Nav.Link as={Link} eventKey="/instances" to="/instances">
                  Instances
                </Nav.Link>
              </NavItem>
              <NavItem href="/explanation">
                <Nav.Link as={Link} eventKey="/explanation" to="/explanation">
                  Explanation
                </Nav.Link>
              </NavItem>
            </Nav>
          </Navbar.Collapse>
        </Navbar>

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
      </main>
    </Route>
  )
}

export default App
