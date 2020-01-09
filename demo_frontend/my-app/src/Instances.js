import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import { Redirect } from "react-router-dom"
import Button from "react-bootstrap/Button"
import {
  MyTable,
  makeColumns,
  makeClasses,
  makeInstances
} from "./MyTableFunctions"

function Instances({ setInstance }) {
  const [response, setResponse] = useState({})
  const [analysis_type, setAnalysis_type] = useState(null)
  const [toAnalyses, setToAnalyses] = useState(false)
  const [selectedClass, setSelectedClass] = useState(null)
  const [selectedInstance, setSelectedInstance] = useState(null)

  const domain = React.useMemo(() => makeColumns(response.domain || []), [
    response.domain
  ])
  const instances = React.useMemo(
    () =>
      Object.entries(response).length === 0 ? [] : makeInstances(response),
    [response]
  )
  const classes = React.useMemo(() => makeClasses(response.classes || []), [
    response.classes
  ])

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/instances")
      const json = await res.json()
      setResponse(json)
      setAnalysis_type(json.analysis_type)
    }

    fetchData()
  }, [])

  const buttondict = {
    explain: "Get Explanation",
    whatif: "What If Analysis",
    mispredicted: "Get Explanation",
    user_rules: "User Rule Analysis",
    explaination_comparison: "Proceed",
    t_class_comparison: "Get Explanations"
  }

  function postInstance(instanceId, class_) {
    setInstance(instanceId)
    return async () => {
      await fetch(`http://127.0.0.1:5000/instance/${instanceId}`, {
        method: "POST",
        body: JSON.stringify({ class: class_ })
      })
      setToAnalyses(true)
    }
  }

  if (response.length === 0) {
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
    switch (analysis_type) {
      case "explain":
        return <Redirect to="/explanation" />

      case "whatif":
        return <Redirect to="/whatif" />

      case "mispredicted":
        return <Redirect to="/explanation" />

      case "user_rules":
        return <Redirect to="/user_rules" />

      case "explaination_comparison":
        return <Redirect to="/classifiers_2" />

      case "t_class_comparison":
        return <Redirect to="/explanation_class_comparison" />

      default:
        return <Redirect to="/explanation" />
    }
  }
  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <h2 className="p-2">Instances</h2>
        <Button
          disabled={selectedInstance === null || selectedClass === null}
          className="ml-auto p-2"
          variant="dark"
          onClick={postInstance(selectedInstance, selectedClass)}
        >
          {buttondict[analysis_type]}
        </Button>
      </Row>
      <Row>
        <Col lg={9}>
          <h2>Select an instance</h2>
          <MyTable
            columns={domain}
            data={instances}
            onCheck={row => e => setSelectedInstance(row.values.id)}
            isChecked={row => row.values.id === selectedInstance}
          />
        </Col>
        <Col lg={3}>
          <h2>Select a class</h2>
          <MyTable
            columns={[
              {
                Header: "Type",
                accessor: "type"
              }
            ]}
            data={classes}
            onCheck={row => e => setSelectedClass(row.values.type)}
            isChecked={row => row.values.type === selectedClass}
          />
        </Col>
      </Row>
    </Container>
  )
}

export default Instances
