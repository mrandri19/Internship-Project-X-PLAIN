import React, {useState, useEffect} from "react"
import {Redirect} from "react-router-dom"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import ListGroup from "react-bootstrap/ListGroup"
import Octicon, {Question, MortarBoard, PrimitiveDot} from "@primer/octicons-react"

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
            {Object.entries(analyses).map(([id, {display_name}]) => (
              <ListGroup.Item
                className="text-center"
                action
                key={id}
                onClick={postAnalysis(id)}
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
                  })(id)}
                />{" "}
                {display_name}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
    </Container>
  )
}

export default Analyses
