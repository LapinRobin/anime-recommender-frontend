$(document).ready(function() {
    var suggestionData = []; // To store suggestion data across eventts

    $("input[name='Mod_name']").on('input', function() {
        var inputVal = $(this).val();
        if (inputVal.length > 1) { // Check to reduce unnecessary requests
            var isSuggestion = suggestionData.includes(inputVal);
            if (isSuggestion) {
                window.location.href = '/description?Mod_name=' + encodeURIComponent(inputVal);
            }
            $.ajax({
                url: "/autocomplete",
                type: "GET",
                dataType: "json",
                data: { term: inputVal },
                success: function(data) {
                    suggestionData = data; // Store data for use later
                    var dataList = $('#anime-suggestions');
                    dataList.empty();
                    $.each(data, function(key, value) {
                        var option = $('<option></option>').attr("value", value);
                        dataList.append(option);
                    });
                }
            });
        }
    });

    $("form").on('submit', function(event) {
        event.preventDefault(); // Prevent the normal form submission
        var inputVal = $("input[name='Mod_name']").val();
        var isSuggestion = suggestionData.includes(inputVal);
        if (isSuggestion) {
            window.location.href = '/description?Mod_name=' + encodeURIComponent(inputVal);
        } else {
            //alert("Please select a valid suggestion from the list.");
            window.location.href = '/search?Mod_name=' + encodeURIComponent(inputVal);
        }
    });

});