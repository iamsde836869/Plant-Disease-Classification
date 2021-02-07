let express = require('express')
let app = express()

app.use(function(req, res, next){
    console.log(`${req.method} request for ${req.url}`);
    next();
})

app.use(express.static("../assets"));

app.listen(2000, function() {
    console.log(`App upto the port 2000`);
})
