import {useTable, useSortBy, usePagination} from "react-table"
import Table from "react-bootstrap/Table"
import Button from "react-bootstrap/Button"
import React, {useState, useEffect} from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import {Redirect} from "react-router-dom"

function Instances() {
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

export default Instances
