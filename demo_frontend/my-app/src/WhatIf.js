import React, {useState, useEffect} from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Button from "react-bootstrap/Button"
import Table from "react-bootstrap/Table"
import Dropdown from "react-bootstrap/Dropdown"

import Rules from "./Rules"
import {ExplanationPlot, getTrace, getDifferences, getNames} from "./ExplanationPlot"

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
            (<Button className="ml-auto p-2"
                     onClick={handleRecompute}>Recompute</Button>)
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
          <p>
            The instance <code>{whatIfExplanation.instance_id}</code> belongs to the
            class <b>{whatIfExplanation.target_class}</b> with probability{" "}
            <code>{whatIfExplanation.prob.toFixed(3)}</code>.
          </p>
          <p>
            The method has converged with error{" "}
            <code>{whatIfExplanation.error.toFixed(3)}</code> and a locality of size{" "}
            <code>{whatIfExplanation.k}</code> (parameter K).
          </p>
          <Rules explanation={whatIfExplanation}/>
        </Col>
      </Row>
    </Container>
  )
}

export default WhatIf
