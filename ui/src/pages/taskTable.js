import React from 'react';
import PropTypes from 'prop-types';
import { withStyles, makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';

import API from '../api';

const StyledTableCell = withStyles(theme => ({
    head: {
      backgroundColor: theme.palette.common.black,
      color: theme.palette.common.white,
    },
    body: {
      fontSize: 14,
    },
  }))(TableCell);

const StyledTableRow = withStyles(theme => ({
  root: {
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.background.default,
    },
  },
}))(TableRow);


const styles = makeStyles(theme => ({
  root: {
    width: '100%',
    marginTop: theme.spacing(3),
    overflowX: 'auto',
  },
  table: {
    minWidth: 700,
  },
}));

const statusLabels={
    "sampling_complete":"Configurations generated. Building TF Models...",
    "fm_complete":"Base FM generated. Sampling configurations...",
    "generation_complete":"Complete. TF models built and trained.",
    "init":"Building Base Feature Model..."
}

class TableHomeComponent extends React.Component {

    constructor(props) {
        // Required step: always call the parent class' constructor
        super(props);

        this.state = {
            headers : [],
            tasks: []

        }

        this.load_tasks()
        setInterval(() => {this.load_tasks()}, 5000);
  
    }

    async load_tasks(){
        API.get(`sample/`, {userId:this.props.usr.id})
          .then(res => {
              var headers = []
              var tasks = []
              headers = ["task_name","formattedTime","formattedStatus", "pdt","products", "nb_initial_config", "nb_valid_elements","max_sampling_time", "nb_training_iterations"]
                
              if(res.data && res.data.length){
                
                //headers = Object.keys(res.data[0])
                tasks = res.data.map(e => {e["formattedStatus"] = statusLabels[e["status"]]; e["formattedTime"] = new Date(parseInt(e["timestamp"])*1000).toLocaleString('fr-FR'); return e;})
                //alert(JSON.stringify(this.state))
        
            }
            this.setState({tasks:tasks, headers:headers})
            
            
        })
    }


    render() {
        const { classes } = this.props;
        const {tasks, headers} = this.state;
    
        return (
                <Paper className={classes.root}>
                <Table className={classes.table}>
                    <TableHead>
                    <TableRow>
                    {headers.map(header => (
                        <StyledTableCell>{header}</StyledTableCell>
                    ))}
                    </TableRow>
                    </TableHead>
                    <TableBody>
                    {tasks.map(row => (
                        <StyledTableRow key={row.id}>
                        {headers.map(val => (
                            <StyledTableCell>{row[val]}</StyledTableCell>
                        ))}
                        </StyledTableRow>
                    ))}
                    </TableBody>
                </Table>
                </Paper>
            );
    }
}

TableHomeComponent.propTypes = {
    classes: PropTypes.object.isRequired
};


const TableHomePage = withStyles(styles)(TableHomeComponent);

export default TableHomePage