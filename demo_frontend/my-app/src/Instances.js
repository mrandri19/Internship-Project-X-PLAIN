import {useTable, useSortBy, usePagination} from "react-table"
import Table from "react-bootstrap/Table"
import React, {useState, useEffect} from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Form from "react-bootstrap/Form"
import {Redirect} from "react-router-dom"
import Button from "react-bootstrap/Button"
import ButtonGroup from "react-bootstrap/ButtonGroup"

function Instances() {
  function MyTable({columns, data, onCheck, isChecked}) {
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
                  <Form.Check checked={isChecked(row)} type="radio"
                              onChange={onCheck(row)}/>
                </td>
                {row.cells.map(cell => {
                  return <td {...cell.getCellProps()}>{cell.render("Cell")}</td>
                })}
              </tr>
            )
          })}
          </tbody>
        </Table>
        <div>
          <div>
            <ButtonGroup className={"mr-3"}>
              <Button onClick={() => gotoPage(0)} disabled={!canPreviousPage}>
                {"<<"}
              </Button>
              {" "}
              <Button onClick={() => previousPage()} disabled={!canPreviousPage}>
                {"<"}
              </Button>
              {" "}
              <Button onClick={() => nextPage()} disabled={!canNextPage}>
                {">"}
              </Button>
              {" "}
              <Button onClick={() => gotoPage(pageCount - 1)} disabled={!canNextPage}>
                {">>"}
              </Button>
            </ButtonGroup>

            <span>
              Page{" "}
              <strong>
                {pageIndex + 1} of {pageOptions.length}
              </strong>{" "}
            </span>
          </div>
          <div className={"mt-3 mb-3"}>

            <Form.Row>
              <Col>
                <Form.Group>
                  <Form.Control
                    size={"sm"}
                    type="number"
                    defaultValue={pageIndex + 1}
                    onChange={e => {
                      const page = e.target.value ? Number(e.target.value) - 1 : 0
                      gotoPage(page)
                    }}
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Control
                    size={"sm"} as={"select"}
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
                  </Form.Control>
                </Form.Group>
              </Col>
            </Form.Row>
          </div>
        </div>
      </>
    )
  }

  function makeInstances(response) {
    return response.instances.map(instance => {
      const row = {}
      row["id"] = instance[1]
      response.domain.forEach((attribute, attribute_ix) => {
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

  function makeClasses(classes) {
    return classes.map(c => {
      return {
        type: c
      }
    })
  }

  const [response, setResponse] = useState({})
  const [toAnalyses, setToAnalyses] = useState(false)
  const [selectedClass, setSelectedClass] = useState(null)
  const [selectedInstance, setSelectedInstance] = useState(null)

  const domain = React.useMemo(() => makeColumns(response.domain || []), [
    response.domain
  ])
  const instances = React.useMemo(
    () => (Object.entries(response).length === 0 ? [] : makeInstances(response)),
    [response]
  )
  const classes = React.useMemo(() => makeClasses(response.classes || []), [response.classes])

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/instances")
      const json = await res.json()
      setResponse(json)
    }

    fetchData()
  }, [])

  function postInstance(instanceId, class_) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/instance/${instanceId}`, {
        method: "POST",
        body: JSON.stringify({"class": class_})
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
      <Row className="mt-3 d-flex align-items-center">
        <h2 className="p-2">Instances</h2>
        <Button disabled={(selectedInstance === null) || (selectedClass === null)}
                className="ml-auto p-2"
                onClick={postInstance(selectedInstance, selectedClass)}>Analyse</Button>
      </Row>
      <Row>
        <Col lg={8}>
          <h2>Select an instance</h2>
          <MyTable columns={domain} data={instances}
                   onCheck={row => e => setSelectedInstance(row.values.id)}
                   isChecked={row => row.values.id === selectedInstance}/>
        </Col>
        <Col lg={4}>
          <h2>Select a class</h2>
          <MyTable columns={[{
            Header: "Type",
            accessor: "type"
          }]} data={classes} onCheck={row => e => setSelectedClass(row.values.type)}
                   isChecked={row => row.values.type === selectedClass}/>
        </Col>
      </Row>
    </Container>
  )
}

export default Instances
