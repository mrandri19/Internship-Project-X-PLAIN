import React, {useState, useEffect} from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"

import Rules from "./Rules"
import {ExplanationPlot, getTrace, getDifferences, getNames} from "./ExplanationPlot"


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

export default Explanation