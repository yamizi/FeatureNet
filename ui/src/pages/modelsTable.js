import React from 'react';
import PropTypes from 'prop-types';
import { withStyles, makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import GetApp from '@material-ui/icons/GetApp';
import VisibilityIcon from '@material-ui/icons/Visibility';

import IconButton from '@material-ui/core/IconButton';

import API from '../api';

const FileDownload = require('js-file-download');

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
  margin: {
    margin: theme.spacing(1),
  },
}));


class TableModelsComponent extends React.Component {

    constructor(props) {
        // Required step: always call the parent class' constructor
        super(props);

        this.state = {
            headers : ["name","nb_blocks","nb_layers","nb_params","robustness","accuracy"],
            task_id:props.task_id,
            models: props.models,
        }
  
    }

    handleDownloadModelClickOpen = (model) => {
      window.open(API.defaults.baseURL+"sample/"+this.state.task_id+"/product/"+model.name+"/model","_blank")
    };

    
    handleDetailsModelClickOpen = (model) => {
      window.open(API.defaults.baseURL+"sample/"+this.state.task_id+"/product/"+model.name+"/graph","_blank")
    };

    render() {
        const { classes } = this.props;
        const {models, headers} = this.state;
    
        return (
                <Paper className={classes.root}>
                <Table className={classes.table}>
                    <TableHead>
                      <TableRow>
                      {headers.map(header => (
                          <StyledTableCell>{header}</StyledTableCell>
                      ))}
                      <StyledTableCell></StyledTableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                    {models.map(row => (
                        <StyledTableRow key={row.id}>
                        {headers.map(val => (
                            <StyledTableCell>{row[val]}</StyledTableCell>
                        ))}
                        <StyledTableCell>
                          <IconButton aria-label="download" className={classes.margin} onClick={() => {this.handleDownloadModelClickOpen(row)}}>
                            <GetApp/>
                          </IconButton>
                          <IconButton aria-label="see" className={classes.margin} onClick={() => {this.handleDetailsModelClickOpen(row)}}>
                            <VisibilityIcon/>
                          </IconButton>
                          
                        </StyledTableCell>
                        </StyledTableRow>
                    ))}
                    </TableBody>
                </Table>
                </Paper>
            );
    }
}

TableModelsComponent.propTypes = {
    classes: PropTypes.object.isRequired
};


const TableModelsPage = withStyles(styles)(TableModelsComponent);

export default TableModelsPage