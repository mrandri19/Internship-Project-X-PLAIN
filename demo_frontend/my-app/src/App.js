import "./style.scss"
import React, {useState, useEffect} from "react"
import "whatwg-fetch"
import {Switch, Route, Link, Redirect, useLocation} from "react-router-dom"
import Plot from "react-plotly.js"
import Navbar from "react-bootstrap/Navbar"
import Nav from "react-bootstrap/Nav"
import NavItem from "react-bootstrap/NavItem"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Dropdown from "react-bootstrap/Dropdown"
import ListGroup from "react-bootstrap/ListGroup"
import Button from "react-bootstrap/Button"
import Spinner from "react-bootstrap/Spinner"
import Table from "react-bootstrap/Table"
import Octicon, {
  Question,
  Book,
  Telescope,
  Italic,
  Graph,
  PrimitiveDot,
  MortarBoard
} from "@primer/octicons-react"
import {useTable, usePagination, useSortBy} from "react-table"

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
    return <Redirect to="/classifiers"/>
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
    return <Redirect to="/instances"/>
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

function MyTable({columns, data, postInstance}) {
  // Use the state and functions returned from useTable to build your UI
  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    prepareRow,
    page, // Instead of using 'rows', we'll use page,
    // which has only the rows for the active page

    // The rest of these things are super handy, too ;)
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize,
    state: {pageIndex, pageSize}
  } = useTable(
    {
      columns,
      data,
      initialState: {pageIndex: 0}
    },
    useSortBy,
    usePagination
  )

  // Render the UI for your table
  return (
    <>
      <Table
        {...getTableProps()}
        hover
        style={{
          display: "block",
          overflowX: "auto",
          whiteSpace: "nowrap"
        }}
      >
        <thead>
        {headerGroups.map(headerGroup => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            <th>{""}</th>
            {headerGroup.headers.map(column => (
              <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                {column.render("Header")}
                <span>
                    {column.isSorted
                      ? column.isSortedDesc
                        ? " 🔽"
                        : " 🔼"
                      : ""}
                  </span>
              </th>
            ))}
          </tr>
        ))}
        </thead>
        <tbody {...getTableBodyProps()}>
        {page.map(row => {
          prepareRow(row)
          return (
            <tr {...row.getRowProps()}>
              <td>
                <Button onClick={postInstance(row.values.id)}>Select</Button>
              </td>
              {row.cells.map(cell => {
                return <td {...cell.getCellProps()}>{cell.render("Cell")}</td>
              })}
            </tr>
          )
        })}
        </tbody>
      </Table>
      {/*
        Pagination can be built however you'd like.
        This is just a very basic UI implementation:
      */}
      <div className="pagination">
        <button onClick={() => gotoPage(0)} disabled={!canPreviousPage}>
          {"<<"}
        </button>
        {" "}
        <button onClick={() => previousPage()} disabled={!canPreviousPage}>
          {"<"}
        </button>
        {" "}
        <button onClick={() => nextPage()} disabled={!canNextPage}>
          {">"}
        </button>
        {" "}
        <button onClick={() => gotoPage(pageCount - 1)} disabled={!canNextPage}>
          {">>"}
        </button>
        {" "}
        <span>
          Page{" "}
          <strong>
            {pageIndex + 1} of {pageOptions.length}
          </strong>{" "}
        </span>
        <span>
          | Go to page:{" "}
          <input
            type="number"
            defaultValue={pageIndex + 1}
            onChange={e => {
              const page = e.target.value ? Number(e.target.value) - 1 : 0
              gotoPage(page)
            }}
            style={{width: "100px"}}
          />
        </span>{" "}
        <select
          value={pageSize}
          onChange={e => {
            setPageSize(Number(e.target.value))
          }}
        >
          {[10, 20, 30, 40, 50].map(pageSize => (
            <option key={pageSize} value={pageSize}>
              Show {pageSize}
            </option>
          ))}
        </select>
      </div>
    </>
  )
}

function Instances() {
  function makeData(instances) {
    return instances.instances.map(instance => {
      const row = {}
      row["id"] = instance[1]
      instances.domain.forEach((attribute, attribute_ix) => {
        row[attribute[0]] = attribute[1][instance[0][attribute_ix]]
      })
      return row
    })
  }

  function makeColumns(domain) {
    return [
      {
        Header: "id",
        accessor: "id"
      },
      ...domain.map(attribute => {
        const name = attribute[0]
        return {
          Header: name,
          accessor: name
        }
      })
    ]
  }

  const [instances, setInstances] = useState({})
  const [toAnalyses, setToAnalyses] = useState(false)

  const columns = React.useMemo(() => makeColumns(instances.domain || []), [
    instances.domain
  ])
  const data = React.useMemo(
    () => (Object.entries(instances).length === 0 ? [] : makeData(instances)),
    [instances]
  )

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
            <Spinner animation="border"/>
          </Col>
        </Row>
      </Container>
    )
  }

  if (toAnalyses) {
    return <Redirect to="/analyses"/>
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
          <MyTable columns={columns} data={data} postInstance={postInstance}/>
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
      if (analysisName === "explain") {
        setToExplanation(true)
      }
      if (analysisName === "whatif") {
        setToWhatIf(true)
      }
    }
  }

  if (toExplanation) {
    return <Redirect to="/explanation"/>
  }
  if (toWhatIf) {
    return <Redirect to="/whatif"/>
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
  const [whatIfExplanation, setwhatIfExplanation] = useState(null)
  const [instanceAttributes, setInstanceAttributes] = useState(null)
  const [recomputeLoading, setRecomputeLoading] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/whatIfExplanation")
      const json = await res.json()
      setwhatIfExplanation(json.explanation)
      setInstanceAttributes(json.attributes)
    }

    fetchData()
  }, [])

  function handleRecompute(e) {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/whatIfExplanation", {
        method: "post",
        body: JSON.stringify(instanceAttributes)
      })
      const json = await res.json()
      setwhatIfExplanation(json.explanation)
      setInstanceAttributes(json.attributes)
      setRecomputeLoading(false)
    }

    setRecomputeLoading(true)
    fetchData()
  }

  if (whatIfExplanation === null || instanceAttributes === null) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>What If Analysis</h2>
            <Spinner animation="border"/>
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = getDifferences(whatIfExplanation)
  const names = getNames(whatIfExplanation)
  const trace = getTrace(differences, names)

  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <h2 className="p-2">What If analysis</h2>
        {
          (recomputeLoading) ?
            (<Button className="ml-auto p-2" variant="primary" disabled>
              <Spinner
                as="span"
                size="sm"
                animation="border"
                role="status"
                aria-hidden="true"
              />
              <span className="sr-only">Loading...</span>
            </Button>) :
            (<Button className="ml-auto p-2" onClick={handleRecompute}>Recompute</Button>)
        }
      </Row>
      <Row className="mb-3">
        <Col>
          <Table size="sm">
            <thead>
            <tr>
              <td>Feature</td>
              <td>Values</td>
            </tr>
            </thead>
            <tbody>
            {Object.entries(instanceAttributes).map(([name, {options, value}]) =>
              <tr key={name}>
                <td>{name}</td>
                <td>
                  <Dropdown onSelect={newValue => {
                    const newInstanceAttributes = {
                      ...instanceAttributes
                    }
                    newInstanceAttributes[name] = {
                      ...newInstanceAttributes[name],
                      value: newValue
                    }

                    setInstanceAttributes(newInstanceAttributes)
                  }}>
                    <Dropdown.Toggle id={name}>
                      {value}
                    </Dropdown.Toggle>
                    <Dropdown.Menu>
                      {options.map(o =>
                        <Dropdown.Item eventKey={o}
                                       key={name + o}>{o}</Dropdown.Item>)}
                    </Dropdown.Menu>
                  </Dropdown>
                </td>
              </tr>
            )}
            </tbody>
          </Table>
        </Col>
        <Col>
          <ExplanationPlot trace={trace}/>
          <Rules explanation={whatIfExplanation}/>
        </Col>
      </Row>
    </Container>
  )
}

function ExplanationPlot({trace}) {
  return <Plot
    data={[trace]}
    layout={{
      title: "Rule/Attribute prediction contribution",
      autosize: true,
      yaxis: {
        type: "category",
        automargin: true,
        dtick: 1,
        categoryorder: "total ascending"
      },
      xaxis: {
        title: "Contribution",
        dtick: 0.1,
        ticks: "inside",
        tickangle: 45
      },
      margin: {
        l: 0,
        r: 40,
        t: 40,
        p: 0
      },
      font: {
        family: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"`,
        size: 16
      }
    }}
    config={{displayModeBar: false, responsive: true}}
  />
}

function getNames(explanation) {
  return explanation.domain
  .map(([name,]) => `${name}=${explanation.instance[name].value}`)
  .concat(
    Object.keys(explanation.map_difference).map((_, ix) => `Rule ${ix + 1}`)
  )
}

function getTrace(differences, names) {
  return {
    type: "bar",
    x: differences,
    y: names,
    orientation: "h",
    marker: {
      color: differences.map(x => {
        return [
          "rgb(165,0,38)",
          "rgb(215,48,39)",
          "rgb(244,109,67)",
          "rgb(253,174,97)",
          "rgb(254,224,144)",
          "rgb(255,255,191)",
          "rgb(224,243,248)",
          "rgb(171,217,233)",
          "rgb(116,173,209)",
          "rgb(69,117,180)",
          "rgb(49,54,149)"
        ][(((x + 1) / 2) * 10) | 0]
      }),
      line: {
        width: 1
      }
    }
  }
}

function getDifferences(explanation) {
  return explanation.diff_single.concat(
    Object.values(explanation.map_difference)
  )
}

function Rules({explanation}) {
  const {map_difference, diff_single, domain, instance} = explanation
  return (<>
    {Object.entries(map_difference)
    .map((rule, ix) => [rule, ix])
    .sort(([[, v1]], [[, v2]]) => v1 < v2)
    .map(([[rule, contribution], ix]) => (
      <p key={rule} style={{fontFamily: "serif", fontSize: "1.1rem"}}>
              <span
                style={{
                  background: [
                    "rgb(165,0,38)",
                    "rgb(215,48,39)",
                    "rgb(244,109,67)",
                    "rgb(253,174,97)",
                    "rgb(254,224,144)",
                    "rgb(255,255,191)",
                    "rgb(224,243,248)",
                    "rgb(171,217,233)",
                    "rgb(116,173,209)",
                    "rgb(69,117,180)",
                    "rgb(49,54,149)"
                  ][(((contribution + 1) / 2) * 10) | 0]
                }}
              >
              Rule {ix + 1}
              </span>{" "}
        ={" "}
        {(() => {
          let attribute_indices = rule.split(",")
          attribute_indices.sort((ix_1, ix_2) => {
            return (
              diff_single[ix_1 - 1] <
              diff_single[ix_2 - 1]
            )
          })

          return (
            <span>
              {"{"}
              {attribute_indices
              .map(ix => domain[ix - 1][0])
              .map(attribute_name => `${attribute_name}=${instance[attribute_name].value}`)
              .join(", ")}
              {"}"}
              </span>
          )
        })()}
      </p>
    ))}
  </>)
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
            <Spinner animation="border"/>
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = getDifferences(explanation)

  const names = getNames(explanation)

  const trace = getTrace(differences, names)

  return (
    <Container>
      <Row className="mt-3 mb-3">
        <Col>
          <h2>Explanation</h2>
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
        </Col>
      </Row>
      <Row>
        <Col>
          <Rules explanation={explanation}/>
        </Col>
        <Col>
          <ExplanationPlot trace={trace}/>
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
                  <Octicon icon={Book}/> Datasets
                </Nav.Link>
              </NavItem>
              <NavItem href="/classifiers">
                <Nav.Link as={Link} eventKey="/classifiers" to="/classifiers">
                  <Octicon icon={Telescope}/> Classifiers
                </Nav.Link>
              </NavItem>
              <NavItem href="/instances">
                <Nav.Link as={Link} eventKey="/instances" to="/instances">
                  <Octicon icon={Italic}/> Instances
                </Nav.Link>
              </NavItem>
              <NavItem href="/analyses">
                <Nav.Link as={Link} eventKey="/analyses" to="/analyses">
                  <Octicon icon={Graph}/> Analyses
                </Nav.Link>
              </NavItem>
            </Nav>
          </Navbar.Collapse>
        </Navbar>

        <Switch>
          <Route path="/datasets">
            <Datasets/>
          </Route>

          <Route path="/classifiers">
            <Classifiers/>
          </Route>

          <Route path="/instances">
            <Instances/>
          </Route>

          <Route path="/analyses">
            <Analyses/>
          </Route>

          <Route path="/whatif">
            <WhatIf/>
          </Route>

          <Route path="/explanation">
            <Explanation/>
          </Route>

          <Route exact path="/">
            <Redirect to="/datasets"/>
          </Route>
          <Route component={RouteNotFound}/>
        </Switch>
      </main>
    </Route>
  )
}

export default App
