$(document).ready(function () {
   
    $('#form_values').submit(function() {
        var url = "../app.py";
        var data = $('#form_values').serialize();
       
        $.ajax({
            url: url,
            data: data,
            success: submit_success,
            error: on_error,
            type: "POST"
   
        });
       
        return false;
    });
   
 });  


var submit_success = function (output_values) {
    response = JSON.parse(output_values);
    var c_cof = $("#compressor_coefficient").text(response.compressor_coefficient);
    var t_cof = $("#turbine_coefficient").text(response.turbine_coefficient);
};

var on_error = function () {
    alert("something went wrong");
};
