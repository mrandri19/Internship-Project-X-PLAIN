import React, {useEffect, useState} from "react"
import {Redirect} from "react-router-dom"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import ListGroup from "react-bootstrap/ListGroup"

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

export default Datasets