import React , { Component } from 'react';

import './App.css';

import { BrowserRouter as Router, Route } from "react-router-dom"
import DashboardComponent from './pages/dashboard'

class App extends Component {

  render() {
    return (

      <Router>
        <div className="App">
          <Route path="/" exact component={DashboardComponent} />
        </div>
      </Router>
      
    );
  }
}

export default App;
