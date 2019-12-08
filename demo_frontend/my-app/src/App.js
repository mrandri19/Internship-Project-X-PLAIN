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
  Italic,
  Check,
  Graph,
  PrimitiveDot,
  MortarBoard
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
      <Row className="mt-3">
        <Col>
          <h2>Select a dataset</h2>
        </Col>
      </Row>
      <Row>
        <Col lg={3}>
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
      <Row className="mt-3">
        <Col>
          <h2>Select a classifier</h2>
        </Col>
      </Row>
      <Row>
        <Col lg={3} className="mt-3">
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
  const [toAnalyses, setToAnalyses] = useState(false)

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
      setToAnalyses(true)
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

  if (toAnalyses) {
    return <Redirect to="/analyses" />
  }
  return (
    <Container>
      <Row className="mt-3">
        <Col>
          <h2>Select an instance</h2>
        </Col>
      </Row>
      <Row>
        <Col>
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
                      <Octicon icon={Check} /> Select
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

function Analyses() {
  const [analyses, setAnalyses] = useState([])

  const [toExplanation, setToExplanation] = useState(false)
  const [toWhatIf, setToWhatIf] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/analyses")
      const json = await res.json()
      setAnalyses(json)
    }
    fetchData()
  }, [])

  function postAnalysis(analysisName) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/analysis/${analysisName}`, {
        method: "POST"
      })
      if (analysisName === "explain") {
        setToExplanation(true)
        return
      }
      if (analysisName === "whatif") {
        setToWhatIf(true)
        return
      }
    }
  }

  if (toExplanation) {
    return <Redirect to="/explanation" />
  }
  if (toWhatIf) {
    return <Redirect to="/whatif" />
  }

  return (
    <Container>
      <Row className="mt-3">
        <Col>
          <h2>Select the analysis to perform</h2>
        </Col>
      </Row>
      <Row>
        <Col lg={3}>
          <ListGroup>
            {analyses.map(analysis => (
              <ListGroup.Item
                className="text-center"
                action
                key={analysis.id}
                onClick={postAnalysis(analysis.id)}
              >
                <Octicon
                  icon={(id => {
                    switch (id) {
                      case "explain":
                        return Question

                      case "whatif":
                        return MortarBoard

                      default:
                        return PrimitiveDot
                    }
                  })(analysis.id)}
                />{" "}
                {analysis.display_name}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
    </Container>
  )
}

function WhatIf() {
  return (
    <Container>
      <Row className="mt-3">
        <Col>
          <h2>What If Analysis</h2>
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
          {Object.keys(explanation.map_difference).map((r, ix) => (
            <p key={r}>
              Rule {ix + 1}:{" "}
              {(() => {
                let attributes = r.split(",")
                attributes.sort((a1, a2) => {
                  return (
                    explanation.diff_single[a1 - 1] <
                    explanation.diff_single[a2 - 1]
                  )
                })

                return attributes
                  .map(a => explanation.domain[a - 1][0])
                  .join(", ")
              })()}
            </p>
          ))}
          <p></p>
        </Col>
        <Col>
          <Plot
            data={[trace]}
            layout={{
              title: "Rule/Attribute prediction contribution",
              autosize: true,
              yaxis: {
                type: "category",
                automargin: true,
                categoryorder: "total ascending"
              },
              xaxis: {
                title: "Contribution",
                dtick: 0.05,
                ticks: "inside"
              },
              margin: {
                l: 0,
                r: 0,
                t: 40,
                p: 0
              },
              font: {
                family: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"`,
                size: 16
              }
            }}
            config={{ displayModeBar: false, responsive: true }}
          />
        </Col>
      </Row>
    </Container>
  )
}

function RouteNotFound() {
  return <h1>Route not found</h1>
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
              <NavItem href="/analyses">
                <Nav.Link as={Link} eventKey="/analyses" to="/analyses">
                  <Octicon icon={Graph} /> Analyses
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

          <Route path="/analyses">
            <Analyses />
          </Route>

          <Route path="/whatif">
            <WhatIf />
          </Route>

          <Route path="/explanation">
            <Explanation />
          </Route>

          <Route exact path="/">
            <Redirect to="/datasets" />
          </Route>
          <Route component={RouteNotFound} />
        </Switch>
      </main>
    </Route>
  )
}

export default App
