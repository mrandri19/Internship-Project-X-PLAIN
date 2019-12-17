import "./style.scss"

import "whatwg-fetch"

import React from "react"

import {Link, Redirect, Route, Switch, useLocation} from "react-router-dom"

import Navbar from "react-bootstrap/Navbar"
import Nav from "react-bootstrap/Nav"
import NavItem from "react-bootstrap/NavItem"


import Octicon, {Book, Graph, Italic, Telescope} from "@primer/octicons-react"

import Datasets from "./Datasets"
import Classifiers from "./Classifiers"
import Instances from "./Instances"
import Analyses from "./Analyses"

import Explanation from "./Explanation"
import WhatIf from "./WhatIf"

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
