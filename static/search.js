$(document).ready(function() {
    $("input[name='Mod_name']").on('input', function() {
        let inputVal = $(this).val();
        if (inputVal.length > 1) { // Check to reduce unnecessary requests
            $.ajax({
                url: "/autocomplete",
                type: "GET",
                dataType: "json",
                data: { term: inputVal },
                success: function(data) {
                    let dataList = $('#anime-suggestions');
                    dataList.empty();
                    $.each(data, function(key, value) {
                        let option = $('<option></option>').attr("value", value);
                        dataList.append(option);
                    });
                }
            });
        }
    });

    $("form").on('submit', function(event) {
        //event.preventDefault(); // Prevent the normal form submission
        //let inputVal = $("input[name='Mod_name']").val();
        //window.location.href = '/search?Mod_name=' + encodeURIComponent(inputVal);
        event.preventDefault(); // Prevent the normal form submission
        let action = $(this).attr('action');
         if (action === '/recommendations') {
             let formData = $(this).serialize(); // Serialize form data

            $.post("/recommendations", formData, function(response) {
            // Redirect to recommendations page
            window.location.href = '/recommendations';
            });
         }else{
             let inputVal = $("input[name='Mod_name']").val();
             window.location.href = '/search?Mod_name=' + encodeURIComponent(inputVal);
         }

    });

});