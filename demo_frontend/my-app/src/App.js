import "./style.scss"
import React, { useState, useEffect } from "react"
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
import Octicon, {
  Question,
  Book,
  Telescope,
  Italic
} from "@primer/octicons-react"

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
        <Col lg={3} className="mt-3">
          <h2>Datasets</h2>
          <ListGroup>
            {datasets.map(datasetName => (
              <ListGroup.Item
                className="text-center"
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
        <Col lg={3} className="mt-3">
          <h2>Classifiers</h2>
          <ListGroup>
            {classifiers.map(classifier => (
              <ListGroup.Item
                className="text-center"
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
          <Table
            hover
            style={{
              display: "block",
              overflowX: "auto",
              whiteSpace: "nowrap"
            }}
          >
            <thead>
              <tr>
                <th></th>
                <th>ID</th>
                {instances.domain.map(([name, _values]) => (
                  <th key={name}>{name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {instances.instances.slice(0, 10).map(instance => (
                <tr key={instance[1]}>
                  <td>
                    <Button size="sm" onClick={postInstance(instance[1])}>
                      <Octicon icon={Question} /> Explain
                    </Button>
                  </td>
                  <td>{instance[1]}</td>
                  {instance[0].map((feature, feature_ix) => (
                    <td key={String(instances[1]) + feature_ix}>
                      {instances.domain.map(d => d[1])[feature_ix][feature]}
                    </td>
                  ))}
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

  const names = explanation.domain
    .map(a => a[0])
    .concat(
      Object.keys(explanation.map_difference).map((_, ix) => `Rule ${ix + 1}`)
    )

  const trace = {
    type: "bar",
    x: differences,
    y: names,
    orientation: "h",
    marker: {
      color: differences
    }
  }

  return (
    <Container>
      <Row className="mt-3">
        <Col>
          <h2>Explanation</h2>
        </Col>
      </Row>
      <Row>
        <Col>
          <p>
            The instance <code>{explanation.instance_id}</code> belongs to the
            class <b>{explanation.target_class}</b> with probability{" "}
            <code>{explanation.prob.toFixed(3)}</code>.
          </p>
          <p>
            The method has converged with error{" "}
            <code>{explanation.error.toFixed(3)}</code> and a locality of size{" "}
            <code>{explanation.k}</code> (parameter K).
          </p>
          <ul>
            {Object.keys(explanation.map_difference).map((r, ix) => (
              <li key={r}>
                Rule {ix + 1}:{" "}
                {r
                  .split(",")
                  .map(a => explanation.domain[a][0])
                  .join(", ")}
              </li>
            ))}
          </ul>
          <p></p>
        </Col>
        <Col>
          <Plot
            data={[trace]}
            layout={{
              autosize: true,
              yaxis: {
                type: "category",
                automargin: true,
                categoryorder: "total ascending"
              },
              xaxis: {
                dtick: 0.05,
                ticks: "inside"
              },
              margin: {
                l: 0,
                r: 0,
                t: 0
                // pad: 0
              },
              font: {
                family: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"`,
                size: 14
              }
            }}
            config={{ displayModeBar: false }}
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
                  <Octicon icon={Book} /> Datasets
                </Nav.Link>
              </NavItem>
              <NavItem href="/classifiers">
                <Nav.Link as={Link} eventKey="/classifiers" to="/classifiers">
                  <Octicon icon={Telescope} /> Classifiers
                </Nav.Link>
              </NavItem>
              <NavItem href="/instances">
                <Nav.Link as={Link} eventKey="/instances" to="/instances">
                  <Octicon icon={Italic} /> Instances
                </Nav.Link>
              </NavItem>
              <NavItem href="/explanation">
                <Nav.Link as={Link} eventKey="/explanation" to="/explanation">
                  <Octicon icon={Question} /> Explanation
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
