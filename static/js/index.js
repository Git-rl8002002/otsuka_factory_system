function show_sensor_position_list(){

        $.ajax({
                type:"POST",
                url:"/show_sensor_position_list",
                data:{
                        
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        //$("#computer_user_detail").show(1000).html(res);
                        $("#show_day_month_detail").show(1000).html(res);
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示電腦序號詳細資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_factory_monitor_detail(val){
        var d_position  = val;

        $.ajax({
                type:"POST",
                url:"/show_factory_monitor_detail",
                data:{
                        's_kind':d_position
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        //$("#computer_user_detail").show(1000).html(res);
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示電腦使用者詳細資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_computer_serial_name_list(){

        $.ajax({
                type:"POST",
                url:"/show_computer_serial_name_list",
                data:{
                        
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        //$("#computer_user_detail").show(1000).html(res);
                        $("#show_day_month_detail").show(1000).html(res);
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示電腦序號詳細資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function search_show_computer_user_detail(val){
        var s_number  = val;

        $.ajax({
                type:"POST",
                url:"/search_show_computer_user_detail",
                data:{
                        's_number':s_number
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        //$("#computer_user_detail").show(1000).html(res);
                        $("#show_day_month_detail").show(1000).html(res);
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示電腦使用者詳細資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_computer_user_detail(val){
        var d_name  = val;

        $.ajax({
                type:"POST",
                url:"/show_computer_user_detail",
                data:{
                        'd_name':d_name
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        //$("#computer_user_detail").show(1000).html(res);
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示電腦使用者詳細資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function download_excel(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/download_day_money_excel",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        //$("#show_day_month_detail").show(1000).html(res);
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("下載日當月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_over_traffic_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_over_traffic_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當超里程月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_traffic_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_traffic_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當交通費月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_parking_fee_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_parking_fee_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當停車費月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_tolls_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_tolls_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當過路費月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_trick_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_trick_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當車票月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_taxi_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_taxi_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當計程車月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_stay_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_stay_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當住宿月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_other_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_other_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當其他月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_oil_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_oil_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當油單月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function show_day_month_detail(val){
        var data  = val.split('/');
        var year  = data[0];
        var month = data[1];

        $.ajax({
                type:"POST",
                url:"/show_day_month_detail",
                data:{
                        'year':year,
                        'month':month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#show_day_month_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("顯示日當月報表資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function load_card_reader_list_detail(value){
        var e_name = value;
        
        $.ajax({
                type:"POST",
                url:"/load_card_reader_list_detail",
                data:{
                        'e_name':e_name
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#card_reader_dep_list_detail").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading 部門人員刷卡紀錄資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });

}

function load_card_reader_list(value){
        var dep = value;

        $.ajax({
                type:"POST",
                url:"/load_card_reader_list",
                data:{
                        'dep':dep
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#card_reader_dep_list").show(1000).html(res);

                        $("#card_reader_dep_list_detail").show(1000).html(" ");
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading 部門人員刷卡紀錄資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function load_work_time_data(value){
        var data = value.split('_');
        var b_date = data[0];
        var e_id   = data[1];
        var e_name = data[2];

        $.ajax({
                type:"POST",
                url:"/load_work_time_data",
                data:{
                        'b_date':b_date,
                        'e_id':e_id,
                        'e_name':e_name
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#load_work_time_data").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading 液劑工時紀錄資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function submit_check_member_2(){
       var employee_id   = $('#employee_id').val();
       var employee_name = $('#employee_name').val();
       var check_year    = $('#check_year').val();
       var check_month   = $('#check_month').val();

       var sir_num1_1 = $('#sir_num1_1').val();
       var sir_num1_2 = $('#sir_num1_2').val();
       var sir_num1_3 = $('#sir_num1_3').val();
       var sir_num1_4 = $('#sir_num1_4').val();

       var sir_num2_1 = $('#sir_num2_1').val();
       var sir_num2_2 = $('#sir_num2_2').val();
       var sir_num2_3 = $('#sir_num2_3').val();
       
       var sir_num3_1 = $('#sir_num3_1').val();
       var sir_num3_2 = $('#sir_num3_2').val();
       var sir_num3_3 = $('#sir_num3_3').val();

       var sir_num4_1 = $('#sir_num4_1').val();
       var sir_num4_2 = $('#sir_num4_2').val();
       var sir_num4_3 = $('#sir_num4_3').val();
       var sir_num4_4 = $('#sir_num4_4').val();

       var sir_num5_1 = $('#sir_num5_1').val();
       var sir_num5_2 = $('#sir_num5_2').val();
       var sir_num5_3 = $('#sir_num5_3').val();

       var sir_num6_1 = $('#sir_num6_1').val();
       var sir_num6_2 = $('#sir_num6_2').val();
       var sir_num6_3 = $('#sir_num6_3').val();

       var sir_num7_1 = $('#sir_num7_1').val();
       var sir_num7_2 = $('#sir_num7_2').val();
       var sir_num7_3 = $('#sir_num7_3').val();
       var sir_num7_4 = $('#sir_num7_4').val();

       var sir_num8_1 = $('#sir_num8_1').val();
       var sir_num8_2 = $('#sir_num8_2').val();
       var sir_num8_3 = $('#sir_num8_3').val();
       var sir_num8_4 = $('#sir_num8_4').val();
       var sir_num8_5 = $('#sir_num8_5').val();

       var comment          = $('#comment').val();
       var other_total      = $('#other_total').val();
       var sir_total        = $('#sir_total').val();
       var other_plus_total = $('#other_plus_total').val();
       var final_total      = $('#final_total').val();
       var final_comment    = $('#final_comment').val();
       
       // check sir_num1_1
       if(sir_num1_1.length == 0){
                alert('一.責任及態度 - 是否主動積極 , 主評不能空白 !');
                exit();
        }
        // check sir_num1_2
        else if(sir_num1_2.length == 0){
                alert('一.責任及態度 - 主管交代是否配合 , 主評不能空白 !');
                exit();
        }
        // check sir_num1_3
        else if(sir_num1_3.length == 0){
                alert('一.責任及態度 - 專注力 , 主評不能空白 !');
                exit();
        }
        // check sir_num1_4
        else if(sir_num1_4.length == 0){
                alert('一.責任及態度 - 不浮誇 , 不欺騙 , 主評不能空白 !');
                exit();
        }
        // check sir_num2_1
        else if(sir_num2_1.length == 0){
                alert('二.工作能力 - 對主身工作是否嚴謹 , 主評不能空白 !');
                exit();
        }
        // check sir_num2_2
        else if(sir_num2_2.length == 0){
                alert('二.工作能力 - 對專業技能是否專精 , 主評不能空白 !');
                exit();
        }
        // check sir_num2_3
        else if(sir_num2_3.length == 0){
                alert('二.工作能力 - 對產品與技術是否全盤了解 , 主評不能空白 !');
                exit();
        }
        // check sir_num3_1
        else if(sir_num3_1.length == 0){
                alert('三.學習狀況 - 是否願接受新事務及挑戰 , 主評不能空白 !');
                exit();
        }
        // check sir_num3_2
        else if(sir_num3_2.length == 0){
                alert('三.學習狀況 - 進入新事務的快慢 , 主評不能空白 !');
                exit();
        }
        // check sir_num3_3
        else if(sir_num3_3.length == 0){
                alert('三.學習狀況 - 對學習專業是否認真 , 主評不能空白 !');
                exit();
        }
        // check sir_num4_1
        else if(sir_num4_1.length == 0){
                alert('四.自主行為 - 服裝儀容及衛生管理 , 主評不能空白 !');
                exit();
        }
        // check sir_num4_2
        else if(sir_num4_2.length == 0){
                alert('四.自主行為 - 對品質觀念是否落實 , 主評不能空白 !');
                exit();
        }
        // check sir_num4_3
        else if(sir_num4_3.length == 0){
                alert('四.自主行為 - 對清潔設備是否落實 , 主評不能空白 !');
                exit();
        }
        // check sir_num4_4
        else if(sir_num4_4.length == 0){
                alert('四.自主行為 - 對器材機具是否愛護 , 主評不能空白 !');
                exit();
        }
        // check sir_num5_1
        else if(sir_num5_1.length == 0){
                alert('五.工作狀態 - 成本意識(不浪費) , 主評不能空白 !');
                exit();
        }
        // check sir_num5_2
        else if(sir_num5_2.length == 0){
                alert('五.工作狀態 - 動作是否俐落 , 主評不能空白 !');
                exit();
        }
        // check sir_num5_3
        else if(sir_num5_3.length == 0){
                alert('五.工作狀態 - 對SOP是否遵從 , 主評不能空白 !');
                exit();
        }
        // check sir_num6_1
        else if(sir_num6_1.length == 0){
                alert('六.團隊默契 - 對同儕及上級相處 , 主評不能空白 !');
                exit();
        }
        // check sir_num6_2
        else if(sir_num6_2.length == 0){
                alert('六.團隊默契 - 主動協助 , 主評不能空白 !');
                exit();
        }
        // check sir_num6_3
        else if(sir_num6_3.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check sir_num7_1
        else if(sir_num7_1.length == 0){
                alert('六.團隊默契 - 對同儕及上級相處 , 主評不能空白 !');
                exit();
        }
        // check sir_num7_2
        else if(sir_num7_2.length == 0){
                alert('六.團隊默契 - 主動協助 , 主評不能空白 !');
                exit();
        }
        // check sir_num7_3
        else if(sir_num7_3.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check sir_num7_4
        else if(sir_num7_4.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check sir_num8_1
        else if(sir_num8_1.length == 0){
                alert('六.團隊默契 - 對同儕及上級相處 , 主評不能空白 !');
                exit();
        }
        // check sir_num8_2
        else if(sir_num8_2.length == 0){
                alert('六.團隊默契 - 主動協助 , 主評不能空白 !');
                exit();
        }
        // check sir_num8_3
        else if(sir_num8_3.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check sir_num8_4
        else if(sir_num8_4.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check sir_num8_5
        else if(sir_num8_5.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check comment
        else if(comment.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        // check final_comment
        else if(final_comment.length == 0){
                alert('六.團隊默契 - 溝通協調 , 主評不能空白 !');
                exit();
        }
        
        else{

                 $.ajax({
                        type:"POST",
                        url:"/update_submit_check_member_2",
                        data:{
                                'employee_id':employee_id,
                                'employee_name':employee_name,
                                'check_year':check_year,
                                'check_month':check_month,
                                'sir_num1_1':sir_num1_1,
                                'sir_num1_2':sir_num1_2,
                                'sir_num1_3':sir_num1_3,
                                'sir_num1_4':sir_num1_4,
                                'sir_num2_1':sir_num2_1,
                                'sir_num2_2':sir_num2_2,
                                'sir_num2_3':sir_num2_3,
                                
                                'sir_num3_1':sir_num3_1,
                                'sir_num3_2':sir_num3_2,
                                'sir_num3_3':sir_num3_3,

                                'sir_num4_1':sir_num4_1,
                                'sir_num4_2':sir_num4_2,
                                'sir_num4_3':sir_num4_3,
                                'sir_num4_4':sir_num4_4,

                                'sir_num5_1':sir_num5_1,
                                'sir_num5_2':sir_num5_2,
                                'sir_num5_3':sir_num5_3,

                                'sir_num6_1':sir_num6_1,
                                'sir_num6_2':sir_num6_2,
                                'sir_num6_3':sir_num6_3,

                                'sir_num7_1':sir_num7_1,
                                'sir_num7_2':sir_num7_2,
                                'sir_num7_3':sir_num7_3,
                                'sir_num7_4':sir_num7_4,

                                'sir_num8_1':sir_num8_1,
                                'sir_num8_2':sir_num8_2,
                                'sir_num8_3':sir_num8_3,
                                'sir_num8_4':sir_num8_4,
                                'sir_num8_5':sir_num8_5,
                                
                                'comment':comment,
                                'sir_total':sir_total,
                                'other_total':other_total,
                                'other_plus_total':other_plus_total,
                                'final_total':final_total,
                                'final_comment':final_comment
                                
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
                                alert(xhr.status);
                                alert(xhr.responseText);
                                alert(throwError);
                                alert(ajaxError);
                        },
                        success:function(res){
                                
                                $("#load_check_form2").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("save 主官評分 考核表資料 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });
        }
}

function load_check_member_list(){
        var employee_id         = $('#employee_id').val();
        var employee_name       = $('#employee_name').val();
        var check_year          = $('#check_year').val();
        var check_month         = $('#check_month').val();

        $.ajax({
                type:"POST",
                url:"/load_check_member_self_list",
                data:{
                        'employee_id':employee_id,
                        'employee_name':employee_name,
                        'check_year':check_year,
                        'check_month':check_month
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#load_check_member_list").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("search 員工姓名資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function department_name_search(){
        var search_name = $('#department_name').val();

        $.ajax({
                type:"POST",
                url:"/department_no_search_val",
                data:{
                        'search_name':search_name
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#search_val").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("search 員工姓名資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
        
}

function submit_work_check_member(){
        var employee_id      = $('#employee_id').val();
        var employee_name    = $('#employee_name').val();
        var department_id    = $('#department_id').val();
        var department_name  = $('#department_name').val();
        var job_title        = $('#job_title').val();
        var b_date           = $('#b_date').val();
        var end_date         = $('#end_date').val();
        var check_year       = $('#check_year').val();
        var check_month      = $('#check_month').val();

        // 1
        var self_num1_1 = $('#self_num1_1').val();
        var self_num1_2 = $('#self_num1_2').val();
        var self_num1_3 = $('#self_num1_3').val();
        var self_num1_4 = $('#self_num1_4').val();
        
        var other_num1_1 = $('#other_num1_1').val();
        var other_num1_2 = $('#other_num1_2').val();
        var other_num1_3 = $('#other_num1_3').val();
        var other_num1_4 = $('#other_num1_4').val();
        
        var sir_num1_1 = $('#sir_num1_1').val();
        var sir_num1_2 = $('#sir_num1_2').val();
        var sir_num1_3 = $('#sir_num1_3').val();
        var sir_num1_4 = $('#sir_num1_4').val();

        // 2
        var self_num2_1 = $('#self_num2_1').val();
        var self_num2_2 = $('#self_num2_2').val();
        var self_num2_3 = $('#self_num2_3').val();
        
        var other_num2_1 = $('#other_num2_1').val();
        var other_num2_2 = $('#other_num2_2').val();
        var other_num2_3 = $('#other_num2_3').val();
        
        var sir_num2_1 = $('#sir_num2_1').val();
        var sir_num2_2 = $('#sir_num2_2').val();
        var sir_num2_3 = $('#sir_num2_3').val();

        // 3
        var self_num3_1 = $('#self_num3_1').val();
        var self_num3_2 = $('#self_num3_2').val();
        var self_num3_3 = $('#self_num3_3').val();
        
        var other_num3_1 = $('#other_num3_1').val();
        var other_num3_2 = $('#other_num3_2').val();
        var other_num3_3 = $('#other_num3_3').val();
        
        var sir_num3_1 = $('#sir_num3_1').val();
        var sir_num3_2 = $('#sir_num3_2').val();
        var sir_num3_3 = $('#sir_num3_3').val();

        // 4
        var self_num4_1 = $('#self_num4_1').val();
        var self_num4_2 = $('#self_num4_2').val();
        var self_num4_3 = $('#self_num4_3').val();
        var self_num4_4 = $('#self_num4_4').val();
        
        var other_num4_1 = $('#other_num4_1').val();
        var other_num4_2 = $('#other_num4_2').val();
        var other_num4_3 = $('#other_num4_3').val();
        var other_num4_4 = $('#other_num4_3').val();
        
        var sir_num4_1 = $('#sir_num4_1').val();
        var sir_num4_2 = $('#sir_num4_2').val();
        var sir_num4_3 = $('#sir_num4_3').val();
        var sir_num4_4 = $('#sir_num4_3').val();

        // 5
        var self_num5_1 = $('#self_num5_1').val();
        var self_num5_2 = $('#self_num5_2').val();
        var self_num5_3 = $('#self_num5_3').val();
        
        var other_num5_1 = $('#other_num5_1').val();
        var other_num5_2 = $('#other_num5_2').val();
        var other_num5_3 = $('#other_num5_3').val();
        
        var sir_num5_1 = $('#sir_num5_1').val();
        var sir_num5_2 = $('#sir_num5_2').val();
        var sir_num5_3 = $('#sir_num5_3').val();

        // 6
        var self_num6_1 = $('#self_num6_1').val();
        var self_num6_2 = $('#self_num6_2').val();
        var self_num6_3 = $('#self_num6_3').val();
        
        var other_num6_1 = $('#other_num6_1').val();
        var other_num6_2 = $('#other_num6_2').val();
        var other_num6_3 = $('#other_num6_3').val();
        
        var sir_num6_1 = $('#sir_num6_1').val();
        var sir_num6_2 = $('#sir_num6_2').val();
        var sir_num6_3 = $('#sir_num6_3').val();

        // 7
        var sir_num7_1 = $('#sir_num7_1').val();
        var sir_num7_2 = $('#sir_num7_2').val();
        var sir_num7_3 = $('#sir_num7_3').val();
        var sir_num7_3 = $('#sir_num7_3').val();
        // 8
        var sir_num8_1 = $('#sir_num8_1').val();
        var sir_num8_2 = $('#sir_num8_2').val();
        var sir_num8_3 = $('#sir_num8_3').val();
        var sir_num8_3 = $('#sir_num8_3').val();

        var comment = $('#comment').val();

        var  self_total         = $('#self_total').val();
        var  other_total        = $('#other_total').val();
        var  sir_total          = $('#sir_total').val();
        var  other_plus_total   = $('#other_plus_total').val();
        var  final_total        = $('#final_total').val();
        var  final_comment      = $('#final_comment').val();

        
        // check self_num1_1
        if(self_num1_1.length == 0){
                //alert('一.責任及態度 - 是否主動積極 , 自評不能空白 !');
                exit();
        }
        // check self_num1_2
        else if(self_num1_2.length == 0){
                //alert('一.責任及態度 - 主管交代是否配合 , 自評不能空白 !');
                exit();
        }
        // check self_num1_3
        else if(self_num1_3.length == 0){
                //alert('一.責任及態度 - 專注力 , 自評不能空白 !');
                exit();
        }
        // check self_num1_4
        else if(self_num1_4.length == 0){
                //alert('一.責任及態度 - 不浮誇 , 不欺騙 , 自評不能空白 !');
                exit();
        }
        // check self_num2_1
        else if(self_num2_1.length == 0){
                //alert('二.工作能力 - 對自身工作是否嚴謹 , 自評不能空白 !');
                exit();
        }
        // check self_num2_2
        else if(self_num2_2.length == 0){
                //alert('二.工作能力 - 對專業技能是否專精 , 自評不能空白 !');
                exit();
        }
        // check self_num2_3
        else if(self_num2_3.length == 0){
                //alert('二.工作能力 - 對產品與技術是否全盤了解 , 自評不能空白 !');
                exit();
        }
        // check self_num3_1
        else if(self_num3_1.length == 0){
                //alert('三.學習狀況 - 是否願接受新事務及挑戰 , 自評不能空白 !');
                exit();
        }
        // check self_num3_2
        else if(self_num3_2.length == 0){
                //alert('三.學習狀況 - 進入新事務的快慢 , 自評不能空白 !');
                exit();
        }
        // check self_num3_3
        else if(self_num3_3.length == 0){
                //alert('三.學習狀況 - 對學習專業是否認真 , 自評不能空白 !');
                exit();
        }
        // check self_num4_1
        else if(self_num4_1.length == 0){
                //alert('四.自主行為 - 服裝儀容及衛生管理 , 自評不能空白 !');
                exit();
        }
        // check self_num4_2
        else if(self_num4_2.length == 0){
                //alert('四.自主行為 - 對品質觀念是否落實 , 自評不能空白 !');
                exit();
        }
        // check self_num4_3
        else if(self_num4_3.length == 0){
                //alert('四.自主行為 - 對清潔設備是否落實 , 自評不能空白 !');
                exit();
        }
        // check self_num4_4
        else if(self_num4_4.length == 0){
                //alert('四.自主行為 - 對器材機具是否愛護 , 自評不能空白 !');
                exit();
        }
        // check self_num5_1
        else if(self_num5_1.length == 0){
                //alert('五.工作狀態 - 成本意識(不浪費) , 自評不能空白 !');
                exit();
        }
        // check self_num5_2
        else if(self_num5_2.length == 0){
                //alert('五.工作狀態 - 動作是否俐落 , 自評不能空白 !');
                exit();
        }
        // check self_num5_3
        else if(self_num5_3.length == 0){
                //alert('五.工作狀態 - 對SOP是否遵從 , 自評不能空白 !');
                exit();
        }
        // check self_num6_1
        else if(self_num6_1.length == 0){
                //alert('六.團隊默契 - 對同儕及上級相處 , 自評不能空白 !');
                exit();
        }
        // check self_num6_2
        else if(self_num6_2.length == 0){
                //alert('六.團隊默契 - 主動協助 , 自評不能空白 !');
                exit();
        }
        // check self_num6_3
        else if(self_num6_3.length == 0){
                //alert('六.團隊默契 - 溝通協調 , 自評不能空白 !');
                exit();
        }
        else if(job_title == '經理') {
                //alert(self_total);
                exit();
        }
        else if(job_title == '課長') {
                //alert(self_total);
                exit();
        }
        else{
                $.ajax({
                        type:"POST",
                        url:"/submit_add_check_member_data",
                        data:{
                                'employee_id':employee_id,
                                'employee_name':employee_name,
                                'department_id':department_id,
                                'department_name':department_name,
                                'job_title':job_title,
                                'b_date':b_date,
                                'end_date':end_date,
                                'check_year':check_year,
                                'check_month':check_month,
                                'self_num1_1':self_num1_1,
                                'self_num1_2':self_num1_2,
                                'self_num1_3':self_num1_3,
                                'self_num1_4':self_num1_4,
                                'self_num2_1':self_num2_1,
                                'self_num2_2':self_num2_2,
                                'self_num2_3':self_num2_3,
                                'self_num3_1':self_num3_1,
                                'self_num3_2':self_num3_2,
                                'self_num3_3':self_num3_3,
                                'self_num4_1':self_num4_1,
                                'self_num4_2':self_num4_2,
                                'self_num4_3':self_num4_3,
                                'self_num4_4':self_num4_4,
                                'self_num5_1':self_num5_1,
                                'self_num5_2':self_num5_2,
                                'self_num5_3':self_num5_3,
                                'self_num6_1':self_num6_1,
                                'self_num6_2':self_num6_2,
                                'self_num6_3':self_num6_3,
                                'self_total':self_total
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
                                alert(xhr.status);
                                alert(xhr.responseText);
                                alert(throwError);
                                alert(ajaxError);
                        },
                        success:function(res){
                                
                                $("#load_check_form2").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading 新增考核人員表單資料 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });
        }
        

}

function submit_add_check_account(){
        
        var employee_id     = $('#employee_id').val()
        var employee_name   = $('#employee_name').val()
        var login_id        = $('#login_id').val()
        var mobile          = $('#mobile').val()
        var department_name = $('#department_name').val()
        var department_code = $('#department_code').val()
        var compyany_id     = $('#company_id').val()
        var end_date        = $('#end_date').val()
        
        
        // check employee_id
        if(employee_id.length == 0){
        //        alert('工號不能空白 !');
                exit();        
        }
        // check employee_name
        else if(employee_name.length == 0){
                //alert('姓名不能空白 !');
                exit();        
        }
        // check login_id
        else if(login_id.length == 0){
                //alert('帳號不能空白 !');
                exit();        
        }
        // check mobile
        else if(mobile.length == 0){
                //alert('密碼不能空白 !');
                exit();        
        }
        // check end_date
        else if(end_date.length == 0){
                //alert('到職不能空白 !');
                exit();        
        }
        else{
                $.ajax({
                        type:"POST",
                        url:"/submit_add_check_account",
                        data:{
                                'employee_id':employee_id,
                                'employee_name':employee_name,
                                'login_id':login_id,
                                'mobile':mobile,
                                'department_name':department_name,
                                'department_code':department_code,
                                'company_id':compyany_id,
                                'end_date':end_date
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
                                alert(xhr.status);
                                alert(xhr.responseText);
                                alert(throwError);
                                alert(ajaxError);
                        },
                        success:function(res){
                                
                                $("#load_check_form").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading 新增考核人員帳號表單資料 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });
        }
        
}

function add_check_member_account(){
        $.ajax({
                type:"POST",
                url:"/add_check_member_account",
                data:{
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#load_check_form").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading 新增考核人員帳號表單 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function load_account_data(user){
        
        $.ajax({
                type:"POST",
                url:"/load_account_data",
                data:{
                        'user':user
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#load_check_form").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        //goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading " + user + " 考核資料 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function auto_plus_other_plus_num(){
        var sir_num1       = $('#sir_num1_1').val();
        var sir_num2       = $('#sir_num1_2').val();
        var sir_num3       = $('#sir_num1_3').val();
        var sir_num4       = $('#sir_num1_4').val();
        var sir_num5       = $('#sir_num2_1').val();
        var sir_num6       = $('#sir_num2_2').val();
        var sir_num7       = $('#sir_num2_3').val();
        var sir_num8       = $('#sir_num3_1').val();
        var sir_num9       = $('#sir_num3_2').val();
        var sir_num10       = $('#sir_num3_2').val();
        var sir_num11       = $('#sir_num4_1').val();
        var sir_num12       = $('#sir_num4_2').val();
        var sir_num13       = $('#sir_num4_3').val();
        var sir_num14       = $('#sir_num4_4').val();
        var sir_num15       = $('#sir_num5_1').val();
        var sir_num16       = $('#sir_num5_2').val();
        var sir_num17       = $('#sir_num5_3').val();
        var sir_num18       = $('#sir_num6_1').val();
        var sir_num19       = $('#sir_num6_2').val();
        var sir_num20       = $('#sir_num6_3').val();
       
        // auto plus normal work time
        var auto_plus_sir_number = (Number(sir_num1) + Number(sir_num2) + Number(sir_num3)  + Number(sir_num4)  + Number(sir_num5)  + Number(sir_num6) + Number(sir_num7) + Number(sir_num8) + Number(sir_num9)  + Number(sir_num10) + Number(sir_num11) + Number(sir_num12) + Number(sir_num13) + Number(sir_num14) + Number(sir_num15) + Number(sir_num16) + Number(sir_num17) + Number(sir_num18) + Number(sir_num19) + Number(sir_num20)).toFixed(0);
        
        var other_plus_num1       = $('#sir_num7_1').val();
        var other_plus_num2       = $('#sir_num7_2').val();
        var other_plus_num3       = $('#sir_num7_3').val();
        var other_plus_num4       = $('#sir_num7_4').val();
        var other_plus_num5       = $('#sir_num8_1').val();
        var other_plus_num6       = $('#sir_num8_2').val();
        var other_plus_num7       = $('#sir_num8_3').val();
        var other_plus_num8       = $('#sir_num8_4').val();
       
        // auto plus normal work time
        var auto_plus_other_plus_number = Number(other_plus_num1) + Number(other_plus_num2) + Number(other_plus_num3) + Number(other_plus_num4) + Number(other_plus_num5) + Number(other_plus_num6) + Number(other_plus_num7) + Number(other_plus_num8);
        
        $('#other_plus_total').val(auto_plus_other_plus_number);
        $('#sir_total').val(auto_plus_sir_number);
        
        // final plus num total
        var final_plus_num_total_val = Number(auto_plus_sir_number) + Number(auto_plus_other_plus_number)
        $('#final_total').val(final_plus_num_total_val);
} 

function auto_plus_sir_num(){
        var sir_num1       = $('#sir_num1_1').val();
        var sir_num2       = $('#sir_num1_2').val();
        var sir_num3       = $('#sir_num1_3').val();
        var sir_num4       = $('#sir_num1_4').val();
        var sir_num5       = $('#sir_num2_1').val();
        var sir_num6       = $('#sir_num2_2').val();
        var sir_num7       = $('#sir_num2_3').val();
        var sir_num8       = $('#sir_num3_1').val();
        var sir_num9       = $('#sir_num3_2').val();
        var sir_num10       = $('#sir_num3_2').val();
        var sir_num11       = $('#sir_num4_1').val();
        var sir_num12       = $('#sir_num4_2').val();
        var sir_num13       = $('#sir_num4_3').val();
        var sir_num14       = $('#sir_num4_4').val();
        var sir_num15       = $('#sir_num5_1').val();
        var sir_num16       = $('#sir_num5_2').val();
        var sir_num17       = $('#sir_num5_3').val();
        var sir_num18       = $('#sir_num6_1').val();
        var sir_num19       = $('#sir_num6_2').val();
        var sir_num20       = $('#sir_num6_3').val();
        
       
        // auto plus normal work time
        var auto_plus_sir_number = (Number(sir_num1) + Number(sir_num2) + Number(sir_num3)  + Number(sir_num4)  + Number(sir_num5)  + Number(sir_num6) + Number(sir_num7) + Number(sir_num8) + Number(sir_num9)  + Number(sir_num10) + Number(sir_num11) + Number(sir_num12) + Number(sir_num13) + Number(sir_num14) + Number(sir_num15) + Number(sir_num16) + Number(sir_num17) + Number(sir_num18) + Number(sir_num19) + Number(sir_num20)).toFixed(0);
        
        var other_plus_num1       = $('#sir_num7_1').val();
        var other_plus_num2       = $('#sir_num7_2').val();
        var other_plus_num3       = $('#sir_num7_3').val();
        var other_plus_num4       = $('#sir_num7_4').val();
        var other_plus_num5       = $('#sir_num8_1').val();
        var other_plus_num6       = $('#sir_num8_2').val();
        var other_plus_num7       = $('#sir_num8_3').val();
        var other_plus_num8       = $('#sir_num8_4').val();
       
        // auto plus normal work time
        var auto_plus_other_plus_number = Number(other_plus_num1) + Number(other_plus_num2) + Number(other_plus_num3) + Number(other_plus_num4) + Number(other_plus_num5) + Number(other_plus_num6) + Number(other_plus_num7) + Number(other_plus_num8);

        $('#other_plus_total').val(auto_plus_other_plus_number);
        $('#sir_total').val(auto_plus_sir_number);
        
        // final plus num total
        var final_plus_num_total_val = Number(auto_plus_sir_number) + Number(auto_plus_other_plus_number)
        $('#final_total').val(final_plus_num_total_val);
} 

function auto_plus_self_num(){
        var self_num1       = $('#self_num1_1').val();
        var self_num2       = $('#self_num1_2').val();
        var self_num3       = $('#self_num1_3').val();
        var self_num4       = $('#self_num1_4').val();
        var self_num5       = $('#self_num2_1').val();
        var self_num6       = $('#self_num2_2').val();
        var self_num7       = $('#self_num2_3').val();
        var self_num8       = $('#self_num3_1').val();
        var self_num9       = $('#self_num3_2').val();
        var self_num10       = $('#self_num3_3').val();
        var self_num11       = $('#self_num4_1').val();
        var self_num12       = $('#self_num4_2').val();
        var self_num13       = $('#self_num4_3').val();
        var self_num14       = $('#self_num4_4').val();
        var self_num15       = $('#self_num5_1').val();
        var self_num16       = $('#self_num5_2').val();
        var self_num17       = $('#self_num5_3').val();
        var self_num18       = $('#self_num6_1').val();
        var self_num19       = $('#self_num6_2').val();
        var self_num20       = $('#self_num6_3').val();
       
        // auto plus normal work time
        var auto_plus_self_number = (Number(self_num1) + Number(self_num2) + Number(self_num3)  + Number(self_num4)  + Number(self_num5)  + Number(self_num6) + Number(self_num7) + Number(self_num8) + Number(self_num9)  + Number(self_num10) + Number(self_num11) + Number(self_num12) + Number(self_num13) + Number(self_num14) + Number(self_num15) + Number(self_num16) + Number(self_num17) + Number(self_num18) + Number(self_num19) + Number(self_num20)).toFixed(0);
        
        $('#self_total').val(auto_plus_self_number);
} 

function submit_work_time_3(){

        // 工號
        var a_work_no = $('#a_work_no').val();                                          
        // 姓名
        var a_name    = $('#a_name').val();                                             
        // 日期
        var a_date    = $('#a_date').val();                                             
        // 稼動工時
        var availability_time = $('#auto_plus_availability_work_time').text();          
        // 加班工時
        var over_time         = $('#auto_plus_over_work_time').text();                  
        // 一般工時
        var normal_tima       = $('#auto_plus_normal_work_time').text();                
        // 總工時
        var total_time        = $('#auto_plus_total_work_time').val();                  
        
        // 第1筆
        var a_work_station_1           = $('#a_work_station_1').val();                  
        var a_production_1             = $('#a_production_1').val();
        var a_product_no_1             = $('#a_product_no_1').val();
        var a_work_normal_time_1       = $('#a_work_normal_time_1').val();
        var a_work_over_time_1         = $('#a_work_over_time_1').val();
        var a_work_availability_time_1 = $('#a_work_availability_time_1').val();
        var a_work_remark_1            = $('#a_work_remark_1').val();

        // 第2筆
        var a_work_station_2           = $('#a_work_station_2').val();                  
        var a_production_2             = $('#a_production_2').val();
        var a_product_no_2             = $('#a_product_no_2').val();
        var a_work_normal_time_2       = $('#a_work_normal_time_2').val();
        var a_work_over_time_2         = $('#a_work_over_time_2').val();
        var a_work_availability_time_2 = $('#a_work_availability_time_2').val();
        var a_work_remark_2            = $('#a_work_remark_2').val();

        // 第3筆
        var a_work_station_3           = $('#a_work_station_3').val();                  
        var a_production_3             = $('#a_production_3').val();
        var a_product_no_3             = $('#a_product_no_3').val();
        var a_work_normal_time_3       = $('#a_work_normal_time_3').val();
        var a_work_over_time_3         = $('#a_work_over_time_3').val();
        var a_work_availability_time_3 = $('#a_work_availability_time_3').val();
        var a_work_remark_3            = $('#a_work_remark_3').val();

        // 第4筆
        var a_work_station_4           = $('#a_work_station_4').val();                  
        var a_production_4             = $('#a_production_4').val();
        var a_product_no_4             = $('#a_product_no_4').val();
        var a_work_normal_time_4       = $('#a_work_normal_time_4').val();
        var a_work_over_time_4         = $('#a_work_over_time_4').val();
        var a_work_availability_time_4 = $('#a_work_availability_time_4').val();
        var a_work_remark_4            = $('#a_work_remark_4').val();

        // 第5筆
        var a_work_station_5           = $('#a_work_station_5').val();                  
        var a_production_5             = $('#a_production_5').val();
        var a_product_no_5             = $('#a_product_no_5').val();
        var a_work_normal_time_5       = $('#a_work_normal_time_5').val();
        var a_work_over_time_5         = $('#a_work_over_time_5').val();
        var a_work_availability_time_5 = $('#a_work_availability_time_5').val();
        var a_work_remark_5            = $('#a_work_remark_5').val();
        
        // 第6筆
        var a_work_station_6           = $('#a_work_station_6').val();                  
        var a_production_6             = $('#a_production_6').val();
        var a_product_no_6             = $('#a_product_no_6').val();
        var a_work_normal_time_6       = $('#a_work_normal_time_6').val();
        var a_work_over_time_6         = $('#a_work_over_time_6').val();
        var a_work_availability_time_6 = $('#a_work_availability_time_6').val();
        var a_work_remark_6            = $('#a_work_remark_6').val();

        // 第7筆
        var a_work_station_7           = $('#a_work_station_7').val();                   
        var a_production_7             = $('#a_production_7').val();
        var a_product_no_7             = $('#a_product_no_7').val();
        var a_work_normal_time_7       = $('#a_work_normal_time_7').val();
        var a_work_over_time_7         = $('#a_work_over_time_7').val();
        var a_work_availability_time_7 = $('#a_work_availability_time_7').val();
        var a_work_remark_7            = $('#a_work_remark_7').val();

        // 第8筆
        var a_work_station_8           = $('#a_work_station_8').val();                   
        var a_production_8             = $('#a_production_8').val();
        var a_product_no_8             = $('#a_product_no_8').val();
        var a_work_normal_time_8       = $('#a_work_normal_time_8').val();
        var a_work_over_time_8         = $('#a_work_over_time_8').val();
        var a_work_availability_time_8 = $('#a_work_availability_time_8').val();
        var a_work_remark_8            = $('#a_work_remark_8').val();
        
        // 第9筆
        var a_work_station_9           = $('#a_work_station_9').val();                   
        var a_production_9             = $('#a_production_9').val();
        var a_product_no_9             = $('#a_product_no_9').val();
        var a_work_normal_time_9       = $('#a_work_normal_time_9').val();
        var a_work_over_time_9         = $('#a_work_over_time_9').val();
        var a_work_availability_time_9 = $('#a_work_availability_time_9').val();
        var a_work_remark_9            = $('#a_work_remark_9').val();

        // 第10筆
        var a_work_station_10           = $('#a_work_station_10').val();                 
        var a_production_10             = $('#a_production_10').val();
        var a_product_no_10             = $('#a_product_no_10').val();
        var a_work_normal_time_10       = $('#a_work_normal_time_10').val();
        var a_work_over_time_10         = $('#a_work_over_time_10').val();
        var a_work_availability_time_10 = $('#a_work_availability_time_10').val();
        var a_work_remark_10            = $('#a_work_remark_10').val();

        // 第11筆
        var a_work_station_11           = $('#a_work_station_11').val();                 
        var a_production_11             = $('#a_production_11').val();
        var a_product_no_11             = $('#a_product_no_11').val();
        var a_work_normal_time_11       = $('#a_work_normal_time_11').val();
        var a_work_over_time_11         = $('#a_work_over_time_11').val();
        var a_work_availability_time_11 = $('#a_work_availability_time_11').val();
        var a_work_remark_11            = $('#a_work_remark_11').val();

        // 第12筆
        var a_work_station_12           = $('#a_work_station_12').val();                 
        var a_production_12             = $('#a_production_12').val();
        var a_product_no_12             = $('#a_product_no_12').val();
        var a_work_normal_time_12       = $('#a_work_normal_time_12').val();
        var a_work_over_time_12         = $('#a_work_over_time_12').val();
        var a_work_availability_time_12 = $('#a_work_availability_time_12').val();
        var a_work_remark_12            = $('#a_work_remark_12').val();

        
        // check 工號
        if(a_work_no.length == 0){
                alert('工號不能空白 !');
                exit();        
        }
        // check 姓名
        else if(a_name == 0){
                alert('姓名不能空白 !');
                exit()
        }
        else{
                alert(  '工號 : ' + a_work_no + ' , 姓名 : ' + a_name + ' , 日期 : ' + a_date + ' , 總工時 : ' + total_time + '\n' +
                        '總一般工時  : ' + normal_tima + ' , 總加班工時 : ' + over_time + ' , 總價動工時 : ' + availability_time + '\n' +
                        a_work_station_1 + ' , ' + a_production_1 + ' , ' + a_product_no_1 + ' , ' + a_work_normal_time_1 + ' , ' + a_work_over_time_1 + ' , ' + a_work_availability_time_1 + ' , ' + a_work_remark_1 + '\n' +
                        a_work_station_2 + ' , ' + a_production_2 + ' , ' + a_product_no_2 + ' , ' + a_work_normal_time_2 + ' , ' + a_work_over_time_2 + ' , ' + a_work_availability_time_2 + ' , ' + a_work_remark_2 + '\n' +
                        a_work_station_3 + ' , ' + a_production_3 + ' , ' + a_product_no_3 + ' , ' + a_work_normal_time_3 + ' , ' + a_work_over_time_3 + ' , ' + a_work_availability_time_3 + ' , ' + a_work_remark_3 + '\n' +
                        a_work_station_4 + ' , ' + a_production_4 + ' , ' + a_product_no_4 + ' , ' + a_work_normal_time_4 + ' , ' + a_work_over_time_4 + ' , ' + a_work_availability_time_4 + ' , ' + a_work_remark_4 + '\n' +
                        a_work_station_5 + ' , ' + a_production_5 + ' , ' + a_product_no_5 + ' , ' + a_work_normal_time_5 + ' , ' + a_work_over_time_5 + ' , ' + a_work_availability_time_5 + ' , ' + a_work_remark_5 + '\n' +
                        a_work_station_6 + ' , ' + a_production_6 + ' , ' + a_product_no_6 + ' , ' + a_work_normal_time_6 + ' , ' + a_work_over_time_6 + ' , ' + a_work_availability_time_6 + ' , ' + a_work_remark_6 + '\n' +
                        a_work_station_7 + ' , ' + a_production_7 + ' , ' + a_product_no_7 + ' , ' + a_work_normal_time_7 + ' , ' + a_work_over_time_7 + ' , ' + a_work_availability_time_7 + ' , ' + a_work_remark_7 + '\n' +
                        a_work_station_8 + ' , ' + a_production_8 + ' , ' + a_product_no_8 + ' , ' + a_work_normal_time_8 + ' , ' + a_work_over_time_8 + ' , ' + a_work_availability_time_8 + ' , ' + a_work_remark_8 + '\n' +
                        a_work_station_9 + ' , ' + a_production_9 + ' , ' + a_product_no_9 + ' , ' + a_work_normal_time_9 + ' , ' + a_work_over_time_9 + ' , ' + a_work_availability_time_9 + ' , ' + a_work_remark_9 + '\n' +
                        a_work_station_10 + ' , ' + a_production_10 + ' , ' + a_product_no_10 + ' , ' + a_work_normal_time_10 + ' , ' + a_work_over_time_10 + ' , ' + a_work_availability_time_10 + ' , ' + a_work_remark_10 + '\n' +
                        a_work_station_11 + ' , ' + a_production_11 + ' , ' + a_product_no_11 + ' , ' + a_work_normal_time_11 + ' , ' + a_work_over_time_11 + ' , ' + a_work_availability_time_11 + ' , ' + a_work_remark_11 + '\n' +
                        a_work_station_12 + ' , ' + a_production_12 + ' , ' + a_product_no_12 + ' , ' + a_work_normal_time_12 + ' , ' + a_work_over_time_12 + ' , ' + a_work_availability_time_12+ ' , ' + a_work_remark_12 + '\n' 
                );
                exit();
        }
        
        $.ajax({
                type:"POST",
                url:"/load_menu_money_record_by_kind",
                data:{
                        'kind':kind
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#menu_money_record_list").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading " + kind + " 種類記帳本清單 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });

}

function submit_work_time_1(){
        
        // 工號
        var a_work_no = $('#a_work_no').val();                                          
        // 姓名
        var a_name    = $('#a_name').val();                                             
        // 日期
        var a_date    = $('#a_date').val();                                             
        // 稼動工時
        var availability_time = $('#auto_plus_availability_work_time').text();          
        // 加班工時
        var over_time         = $('#auto_plus_over_work_time').text();                  
        // 一般工時
        var normal_tima       = $('#auto_plus_normal_work_time').text();                
        // 總工時
        var total_time        = $('#auto_plus_total_work_time').val();                  
        
        // 第1筆
        var a_work_station_1           = $('#a_work_station_1').val();                  
        var a_production_1             = $('#a_production_1').val();
        var a_product_no_1             = $('#a_product_no_1').val();
        var a_work_normal_time_1       = $('#a_work_normal_time_1').val();
        var a_work_over_time_1         = $('#a_work_over_time_1').val();
        var a_work_availability_time_1 = $('#a_work_availability_time_1').val();
        var a_work_remark_1            = $('#a_work_remark_1').val();

        // 第2筆
        var a_work_station_2           = $('#a_work_station_2').val();                  
        var a_production_2             = $('#a_production_2').val();
        var a_product_no_2             = $('#a_product_no_2').val();
        var a_work_normal_time_2       = $('#a_work_normal_time_2').val();
        var a_work_over_time_2         = $('#a_work_over_time_2').val();
        var a_work_availability_time_2 = $('#a_work_availability_time_2').val();
        var a_work_remark_2            = $('#a_work_remark_2').val();

        // 第3筆
        var a_work_station_3           = $('#a_work_station_3').val();                  
        var a_production_3             = $('#a_production_3').val();
        var a_product_no_3             = $('#a_product_no_3').val();
        var a_work_normal_time_3       = $('#a_work_normal_time_3').val();
        var a_work_over_time_3         = $('#a_work_over_time_3').val();
        var a_work_availability_time_3 = $('#a_work_availability_time_3').val();
        var a_work_remark_3            = $('#a_work_remark_3').val();

        // 第4筆
        var a_work_station_4           = $('#a_work_station_4').val();                  
        var a_production_4             = $('#a_production_4').val();
        var a_product_no_4             = $('#a_product_no_4').val();
        var a_work_normal_time_4       = $('#a_work_normal_time_4').val();
        var a_work_over_time_4         = $('#a_work_over_time_4').val();
        var a_work_availability_time_4 = $('#a_work_availability_time_4').val();
        var a_work_remark_4            = $('#a_work_remark_4').val();

        // 第5筆
        var a_work_station_5           = $('#a_work_station_5').val();                  
        var a_production_5             = $('#a_production_5').val();
        var a_product_no_5             = $('#a_product_no_5').val();
        var a_work_normal_time_5       = $('#a_work_normal_time_5').val();
        var a_work_over_time_5         = $('#a_work_over_time_5').val();
        var a_work_availability_time_5 = $('#a_work_availability_time_5').val();
        var a_work_remark_5            = $('#a_work_remark_5').val();
        
        // 第6筆
        var a_work_station_6           = $('#a_work_station_6').val();                  
        var a_production_6             = $('#a_production_6').val();
        var a_product_no_6             = $('#a_product_no_6').val();
        var a_work_normal_time_6       = $('#a_work_normal_time_6').val();
        var a_work_over_time_6         = $('#a_work_over_time_6').val();
        var a_work_availability_time_6 = $('#a_work_availability_time_6').val();
        var a_work_remark_6            = $('#a_work_remark_6').val();

        // 第7筆
        var a_work_station_7           = $('#a_work_station_7').val();                   
        var a_production_7             = $('#a_production_7').val();
        var a_product_no_7             = $('#a_product_no_7').val();
        var a_work_normal_time_7       = $('#a_work_normal_time_7').val();
        var a_work_over_time_7         = $('#a_work_over_time_7').val();
        var a_work_availability_time_7 = $('#a_work_availability_time_7').val();
        var a_work_remark_7            = $('#a_work_remark_7').val();

        // 第8筆
        var a_work_station_8           = $('#a_work_station_8').val();                   
        var a_production_8             = $('#a_production_8').val();
        var a_product_no_8             = $('#a_product_no_8').val();
        var a_work_normal_time_8       = $('#a_work_normal_time_8').val();
        var a_work_over_time_8         = $('#a_work_over_time_8').val();
        var a_work_availability_time_8 = $('#a_work_availability_time_8').val();
        var a_work_remark_8            = $('#a_work_remark_8').val();
        
        // 第9筆
        var a_work_station_9           = $('#a_work_station_9').val();                   
        var a_production_9             = $('#a_production_9').val();
        var a_product_no_9             = $('#a_product_no_9').val();
        var a_work_normal_time_9       = $('#a_work_normal_time_9').val();
        var a_work_over_time_9         = $('#a_work_over_time_9').val();
        var a_work_availability_time_9 = $('#a_work_availability_time_9').val();
        var a_work_remark_9            = $('#a_work_remark_9').val();

        // 第10筆
        var a_work_station_10           = $('#a_work_station_10').val();                 
        var a_production_10             = $('#a_production_10').val();
        var a_product_no_10             = $('#a_product_no_10').val();
        var a_work_normal_time_10       = $('#a_work_normal_time_10').val();
        var a_work_over_time_10         = $('#a_work_over_time_10').val();
        var a_work_availability_time_10 = $('#a_work_availability_time_10').val();
        var a_work_remark_10            = $('#a_work_remark_10').val();

        // 第11筆
        var a_work_station_11           = $('#a_work_station_11').val();                 
        var a_production_11             = $('#a_production_11').val();
        var a_product_no_11             = $('#a_product_no_11').val();
        var a_work_normal_time_11       = $('#a_work_normal_time_11').val();
        var a_work_over_time_11         = $('#a_work_over_time_11').val();
        var a_work_availability_time_11 = $('#a_work_availability_time_11').val();
        var a_work_remark_11            = $('#a_work_remark_11').val();

        // 第12筆
        var a_work_station_12           = $('#a_work_station_12').val();                 
        var a_production_12             = $('#a_production_12').val();
        var a_product_no_12             = $('#a_product_no_12').val();
        var a_work_normal_time_12       = $('#a_work_normal_time_12').val();
        var a_work_over_time_12         = $('#a_work_over_time_12').val();
        var a_work_availability_time_12 = $('#a_work_availability_time_12').val();
        var a_work_remark_12            = $('#a_work_remark_12').val();

        
        // check 工號
        if(a_work_no.length == 0){
                alert('工號不能空白 !');
                exit();        
        }
        // check 姓名
        else if(a_name == 0){
                alert('姓名不能空白 !');
                exit()
        }
        else{
                alert(  '工號 : ' + a_work_no + ' , 姓名 : ' + a_name + ' , 日期 : ' + a_date + ' , 總工時 : ' + total_time + '\n' +
                        '總一般工時  : ' + normal_tima + ' , 總加班工時 : ' + over_time + ' , 總價動工時 : ' + availability_time + '\n' +
                        a_work_station_1 + ' , ' + a_production_1 + ' , ' + a_product_no_1 + ' , ' + a_work_normal_time_1 + ' , ' + a_work_over_time_1 + ' , ' + a_work_availability_time_1 + ' , ' + a_work_remark_1 + '\n' +
                        a_work_station_2 + ' , ' + a_production_2 + ' , ' + a_product_no_2 + ' , ' + a_work_normal_time_2 + ' , ' + a_work_over_time_2 + ' , ' + a_work_availability_time_2 + ' , ' + a_work_remark_2 + '\n' +
                        a_work_station_3 + ' , ' + a_production_3 + ' , ' + a_product_no_3 + ' , ' + a_work_normal_time_3 + ' , ' + a_work_over_time_3 + ' , ' + a_work_availability_time_3 + ' , ' + a_work_remark_3 + '\n' +
                        a_work_station_4 + ' , ' + a_production_4 + ' , ' + a_product_no_4 + ' , ' + a_work_normal_time_4 + ' , ' + a_work_over_time_4 + ' , ' + a_work_availability_time_4 + ' , ' + a_work_remark_4 + '\n' +
                        a_work_station_5 + ' , ' + a_production_5 + ' , ' + a_product_no_5 + ' , ' + a_work_normal_time_5 + ' , ' + a_work_over_time_5 + ' , ' + a_work_availability_time_5 + ' , ' + a_work_remark_5 + '\n' +
                        a_work_station_6 + ' , ' + a_production_6 + ' , ' + a_product_no_6 + ' , ' + a_work_normal_time_6 + ' , ' + a_work_over_time_6 + ' , ' + a_work_availability_time_6 + ' , ' + a_work_remark_6 + '\n' +
                        a_work_station_7 + ' , ' + a_production_7 + ' , ' + a_product_no_7 + ' , ' + a_work_normal_time_7 + ' , ' + a_work_over_time_7 + ' , ' + a_work_availability_time_7 + ' , ' + a_work_remark_7 + '\n' +
                        a_work_station_8 + ' , ' + a_production_8 + ' , ' + a_product_no_8 + ' , ' + a_work_normal_time_8 + ' , ' + a_work_over_time_8 + ' , ' + a_work_availability_time_8 + ' , ' + a_work_remark_8 + '\n' +
                        a_work_station_9 + ' , ' + a_production_9 + ' , ' + a_product_no_9 + ' , ' + a_work_normal_time_9 + ' , ' + a_work_over_time_9 + ' , ' + a_work_availability_time_9 + ' , ' + a_work_remark_9 + '\n' +
                        a_work_station_10 + ' , ' + a_production_10 + ' , ' + a_product_no_10 + ' , ' + a_work_normal_time_10 + ' , ' + a_work_over_time_10 + ' , ' + a_work_availability_time_10 + ' , ' + a_work_remark_10 + '\n' +
                        a_work_station_11 + ' , ' + a_production_11 + ' , ' + a_product_no_11 + ' , ' + a_work_normal_time_11 + ' , ' + a_work_over_time_11 + ' , ' + a_work_availability_time_11 + ' , ' + a_work_remark_11 + '\n' +
                        a_work_station_12 + ' , ' + a_production_12 + ' , ' + a_product_no_12 + ' , ' + a_work_normal_time_12 + ' , ' + a_work_over_time_12 + ' , ' + a_work_availability_time_12+ ' , ' + a_work_remark_12 + '\n' 
                );
                exit();
        }
        
        $.ajax({
                type:"POST",
                url:"/load_menu_money_record_by_kind",
                data:{
                        'kind':kind
                },
                datatype:"html",
                        error:function(xhr , ajaxError , throwError){
                        alert(xhr.status);
                        alert(xhr.responseText);
                        alert(throwError);
                        alert(ajaxError);
                },
                success:function(res){
                        
                        $("#menu_money_record_list").show(1000).html(res);
                        
                        // scroll page bottom to page top
                        goto_top();
                        
                        //location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading " + kind + " 種類記帳本清單 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });

}
   
function submit_work_time(){

        // 工號
        var a_work_no = $('#a_work_no').val();                                          
        // 姓名
        var a_name    = $('#a_name').val();                                             
        // 日期
        var a_date    = $('#a_date').val();                                             
        // 部門代號
        var dep_id    = $('#department_id').val();                                             
        // 稼動工時
        var availability_time = $('#auto_plus_availability_work_time').text();          
        // 加班工時
        var over_time         = $('#auto_plus_over_work_time').text();              
        // 一般工時
        var normal_time       = $('#auto_plus_normal_work_time').text();                
        // 總工時
        var total_time        = $('#auto_plus_total_work_time').val();                  
        
        // 第1筆
        var a_work_station_1           = $('#a_work_station_1').val();                  
        var a_production_1             = $('#a_production_1').val();
        var a_product_no_1             = $('#a_product_no_1').val();
        var a_work_normal_time_1       = $('#a_work_normal_time_1').val();
        var a_work_over_time_1         = $('#a_work_over_time_1').val();
        var a_work_availability_time_1 = $('#a_work_availability_time_1').val();
        var a_work_remark_1            = $('#a_work_remark_1').val();

        // 第2筆
        var a_work_station_2           = $('#a_work_station_2').val();                  
        var a_production_2             = $('#a_production_2').val();
        var a_product_no_2             = $('#a_product_no_2').val();
        var a_work_normal_time_2       = $('#a_work_normal_time_2').val();
        var a_work_over_time_2         = $('#a_work_over_time_2').val();
        var a_work_availability_time_2 = $('#a_work_availability_time_2').val();
        var a_work_remark_2            = $('#a_work_remark_2').val();

        // 第3筆
        var a_work_station_3           = $('#a_work_station_3').val();                  
        var a_production_3             = $('#a_production_3').val();
        var a_product_no_3             = $('#a_product_no_3').val();
        var a_work_normal_time_3       = $('#a_work_normal_time_3').val();
        var a_work_over_time_3         = $('#a_work_over_time_3').val();
        var a_work_availability_time_3 = $('#a_work_availability_time_3').val();
        var a_work_remark_3            = $('#a_work_remark_3').val();

        // 第4筆
        var a_work_station_4           = $('#a_work_station_4').val();                  
        var a_production_4             = $('#a_production_4').val();
        var a_product_no_4             = $('#a_product_no_4').val();
        var a_work_normal_time_4       = $('#a_work_normal_time_4').val();
        var a_work_over_time_4         = $('#a_work_over_time_4').val();
        var a_work_availability_time_4 = $('#a_work_availability_time_4').val();
        var a_work_remark_4            = $('#a_work_remark_4').val();

        // 第5筆
        var a_work_station_5           = $('#a_work_station_5').val();                  
        var a_production_5             = $('#a_production_5').val();
        var a_product_no_5             = $('#a_product_no_5').val();
        var a_work_normal_time_5       = $('#a_work_normal_time_5').val();
        var a_work_over_time_5         = $('#a_work_over_time_5').val();
        var a_work_availability_time_5 = $('#a_work_availability_time_5').val();
        var a_work_remark_5            = $('#a_work_remark_5').val();
        
        // 第6筆
        var a_work_station_6           = $('#a_work_station_6').val();                  
        var a_production_6             = $('#a_production_6').val();
        var a_product_no_6             = $('#a_product_no_6').val();
        var a_work_normal_time_6       = $('#a_work_normal_time_6').val();
        var a_work_over_time_6         = $('#a_work_over_time_6').val();
        var a_work_availability_time_6 = $('#a_work_availability_time_6').val();
        var a_work_remark_6            = $('#a_work_remark_6').val();

        // 第7筆
        var a_work_station_7           = $('#a_work_station_7').val();                   
        var a_production_7             = $('#a_production_7').val();
        var a_product_no_7             = $('#a_product_no_7').val();
        var a_work_normal_time_7       = $('#a_work_normal_time_7').val();
        var a_work_over_time_7         = $('#a_work_over_time_7').val();
        var a_work_availability_time_7 = $('#a_work_availability_time_7').val();
        var a_work_remark_7            = $('#a_work_remark_7').val();

        // 第8筆
        var a_work_station_8           = $('#a_work_station_8').val();                   
        var a_production_8             = $('#a_production_8').val();
        var a_product_no_8             = $('#a_product_no_8').val();
        var a_work_normal_time_8       = $('#a_work_normal_time_8').val();
        var a_work_over_time_8         = $('#a_work_over_time_8').val();
        var a_work_availability_time_8 = $('#a_work_availability_time_8').val();
        var a_work_remark_8            = $('#a_work_remark_8').val();
        
        // 第9筆
        var a_work_station_9           = $('#a_work_station_9').val();                   
        var a_production_9             = $('#a_production_9').val();
        var a_product_no_9             = $('#a_product_no_9').val();
        var a_work_normal_time_9       = $('#a_work_normal_time_9').val();
        var a_work_over_time_9         = $('#a_work_over_time_9').val();
        var a_work_availability_time_9 = $('#a_work_availability_time_9').val();
        var a_work_remark_9            = $('#a_work_remark_9').val();

        // 第10筆
        var a_work_station_10           = $('#a_work_station_10').val();                 
        var a_production_10             = $('#a_production_10').val();
        var a_product_no_10             = $('#a_product_no_10').val();
        var a_work_normal_time_10       = $('#a_work_normal_time_10').val();
        var a_work_over_time_10         = $('#a_work_over_time_10').val();
        var a_work_availability_time_10 = $('#a_work_availability_time_10').val();
        var a_work_remark_10            = $('#a_work_remark_10').val();

        // 第11筆
        var a_work_station_11           = $('#a_work_station_11').val();                 
        var a_production_11             = $('#a_production_11').val();
        var a_product_no_11             = $('#a_product_no_11').val();
        var a_work_normal_time_11       = $('#a_work_normal_time_11').val();
        var a_work_over_time_11         = $('#a_work_over_time_11').val();
        var a_work_availability_time_11 = $('#a_work_availability_time_11').val();
        var a_work_remark_11            = $('#a_work_remark_11').val();

        // 第12筆
        var a_work_station_12           = $('#a_work_station_12').val();                 
        var a_production_12             = $('#a_production_12').val();
        var a_product_no_12             = $('#a_product_no_12').val();
        var a_work_normal_time_12       = $('#a_work_normal_time_12').val();
        var a_work_over_time_12         = $('#a_work_over_time_12').val();
        var a_work_availability_time_12 = $('#a_work_availability_time_12').val();
        var a_work_remark_12            = $('#a_work_remark_12').val();

        
        // check 工號
        if(a_work_station_1.length == 0){
                alert('工號不能空白 !');
                exit();        
        }
        else{
                /*
                alert(  '工號 : ' + a_work_no + ' , 姓名 : ' + a_name + ' , 部門代號 : ' + dep_id + ' , 日期 : ' + a_date + ' , 總工時 : ' + total_time + '\n' +
                        '總一般工時  : ' + normal_time + ' , 總加班工時 : ' + over_time + ' , 總價動工時 : ' + availability_time + '\n' +
                        a_work_station_1 + ' , ' + a_production_1 + ' , ' + a_product_no_1 + ' , ' + a_work_normal_time_1 + ' , ' + a_work_over_time_1 + ' , ' + a_work_availability_time_1 + ' , ' + a_work_remark_1 + '\n' +
                        a_work_station_2 + ' , ' + a_production_2 + ' , ' + a_product_no_2 + ' , ' + a_work_normal_time_2 + ' , ' + a_work_over_time_2 + ' , ' + a_work_availability_time_2 + ' , ' + a_work_remark_2 + '\n' +
                        a_work_station_3 + ' , ' + a_production_3 + ' , ' + a_product_no_3 + ' , ' + a_work_normal_time_3 + ' , ' + a_work_over_time_3 + ' , ' + a_work_availability_time_3 + ' , ' + a_work_remark_3 + '\n' +
                        a_work_station_4 + ' , ' + a_production_4 + ' , ' + a_product_no_4 + ' , ' + a_work_normal_time_4 + ' , ' + a_work_over_time_4 + ' , ' + a_work_availability_time_4 + ' , ' + a_work_remark_4 + '\n' +
                        a_work_station_5 + ' , ' + a_production_5 + ' , ' + a_product_no_5 + ' , ' + a_work_normal_time_5 + ' , ' + a_work_over_time_5 + ' , ' + a_work_availability_time_5 + ' , ' + a_work_remark_5 + '\n' +
                        a_work_station_6 + ' , ' + a_production_6 + ' , ' + a_product_no_6 + ' , ' + a_work_normal_time_6 + ' , ' + a_work_over_time_6 + ' , ' + a_work_availability_time_6 + ' , ' + a_work_remark_6 + '\n' +
                        a_work_station_7 + ' , ' + a_production_7 + ' , ' + a_product_no_7 + ' , ' + a_work_normal_time_7 + ' , ' + a_work_over_time_7 + ' , ' + a_work_availability_time_7 + ' , ' + a_work_remark_7 + '\n' +
                        a_work_station_8 + ' , ' + a_production_8 + ' , ' + a_product_no_8 + ' , ' + a_work_normal_time_8 + ' , ' + a_work_over_time_8 + ' , ' + a_work_availability_time_8 + ' , ' + a_work_remark_8 + '\n' +
                        a_work_station_9 + ' , ' + a_production_9 + ' , ' + a_product_no_9 + ' , ' + a_work_normal_time_9 + ' , ' + a_work_over_time_9 + ' , ' + a_work_availability_time_9 + ' , ' + a_work_remark_9 + '\n' +
                        a_work_station_10 + ' , ' + a_production_10 + ' , ' + a_product_no_10 + ' , ' + a_work_normal_time_10 + ' , ' + a_work_over_time_10 + ' , ' + a_work_availability_time_10 + ' , ' + a_work_remark_10 + '\n' +
                        a_work_station_11 + ' , ' + a_production_11 + ' , ' + a_product_no_11 + ' , ' + a_work_normal_time_11 + ' , ' + a_work_over_time_11 + ' , ' + a_work_availability_time_11 + ' , ' + a_work_remark_11 + '\n' +
                        a_work_station_12 + ' , ' + a_production_12 + ' , ' + a_product_no_12 + ' , ' + a_work_normal_time_12 + ' , ' + a_work_over_time_12 + ' , ' + a_work_availability_time_12+ ' , ' + a_work_remark_12 + '\n' 
                );
                exit();
                */
                
                $.ajax({
                        type:"POST",
                        url:"/submit_work_time",
                        data:{
                                'a_work_no':a_work_no,
                                'a_name':a_name,
                                'dep_id':dep_id,
                                'b_date':a_date,
                                'total_time':total_time,
                                'normal_time':normal_time,
                                'over_time':over_time,
                                'availability_time':availability_time,
                                
                                'a_work_station_1':a_work_station_1, 
                                'a_production_1':a_production_1,
                                'a_product_no_1':a_product_no_1,
                                'a_work_normal_time_1':a_work_normal_time_1,
                                'a_work_over_time_1':a_work_over_time_1,
                                'a_work_availability_time_1':a_work_availability_time_1, 
                                'a_work_remark_1':a_work_remark_1,

                                'a_work_station_2':a_work_station_2, 
                                'a_production_2':a_production_2,
                                'a_product_no_2':a_product_no_2,
                                'a_work_normal_time_2':a_work_normal_time_2,
                                'a_work_over_time_2':a_work_over_time_2,
                                'a_work_availability_time_2':a_work_availability_time_2, 
                                'a_work_remark_2':a_work_remark_2,
                                
                                'a_work_remark_3':a_work_remark_3,
                                'a_work_station_3':a_work_station_3, 
                                'a_production_3':a_production_3,
                                'a_product_no_3':a_product_no_3,
                                'a_work_normal_time_3':a_work_normal_time_3,
                                'a_work_over_time_3':a_work_over_time_3,
                                'a_work_availability_time_3':a_work_availability_time_3, 
                                'a_work_remark_3':a_work_remark_3,

                                'a_work_remark_4':a_work_remark_4,
                                'a_work_station_4':a_work_station_4, 
                                'a_production_4':a_production_4,
                                'a_product_no_4':a_product_no_4,
                                'a_work_normal_time_4':a_work_normal_time_4,
                                'a_work_over_time_4':a_work_over_time_4,
                                'a_work_availability_time_4':a_work_availability_time_4, 
                                'a_work_remark_4':a_work_remark_4,

                                'a_work_remark_5':a_work_remark_5,
                                'a_work_station_5':a_work_station_5, 
                                'a_production_5':a_production_5,
                                'a_product_no_5':a_product_no_5,
                                'a_work_normal_time_5':a_work_normal_time_5,
                                'a_work_over_time_5':a_work_over_time_5,
                                'a_work_availability_time_5':a_work_availability_time_5, 
                                'a_work_remark_5':a_work_remark_5,

                                'a_work_remark_6':a_work_remark_6,
                                'a_work_station_6':a_work_station_6, 
                                'a_production_6':a_production_6,
                                'a_product_no_6':a_product_no_6,
                                'a_work_normal_time_6':a_work_normal_time_6,
                                'a_work_over_time_6':a_work_over_time_6,
                                'a_work_availability_time_6':a_work_availability_time_6, 
                                'a_work_remark_6':a_work_remark_6,

                                'a_work_remark_7':a_work_remark_7,
                                'a_work_station_7':a_work_station_7, 
                                'a_production_7':a_production_7,
                                'a_product_no_7':a_product_no_7,
                                'a_work_normal_time_7':a_work_normal_time_7,
                                'a_work_over_time_7':a_work_over_time_7,
                                'a_work_availability_time_7':a_work_availability_time_7, 
                                'a_work_remark_7':a_work_remark_7,

                                'a_work_remark_8':a_work_remark_8,
                                'a_work_station_8':a_work_station_8, 
                                'a_production_8':a_production_8,
                                'a_product_no_8':a_product_no_8,
                                'a_work_normal_time_8':a_work_normal_time_8,
                                'a_work_over_time_8':a_work_over_time_8,
                                'a_work_availability_time_8':a_work_availability_time_8, 
                                'a_work_remark_8':a_work_remark_8,

                                'a_work_remark_9':a_work_remark_9,
                                'a_work_station_9':a_work_station_9, 
                                'a_production_9':a_production_9,
                                'a_product_no_9':a_product_no_9,
                                'a_work_normal_time_9':a_work_normal_time_9,
                                'a_work_over_time_9':a_work_over_time_9,
                                'a_work_availability_time_9':a_work_availability_time_9, 
                                'a_work_remark_9':a_work_remark_9,

                                'a_work_remark_10':a_work_remark_10,
                                'a_work_station_10':a_work_station_10, 
                                'a_production_10':a_production_10,
                                'a_product_no_10':a_product_no_10,
                                'a_work_normal_time_10':a_work_normal_time_10,
                                'a_work_over_time_10':a_work_over_time_10,
                                'a_work_availability_time_10':a_work_availability_time_10, 
                                'a_work_remark_10':a_work_remark_10,

                                'a_work_remark_11':a_work_remark_11,
                                'a_work_station_11':a_work_station_11, 
                                'a_production_11':a_production_11,
                                'a_product_no_11':a_product_no_11,
                                'a_work_normal_time_11':a_work_normal_time_11,
                                'a_work_over_time_11':a_work_over_time_11,
                                'a_work_availability_time_11':a_work_availability_time_11, 
                                'a_work_remark_11':a_work_remark_11,

                                'a_work_remark_12':a_work_remark_12,
                                'a_work_station_12':a_work_station_12, 
                                'a_production_12':a_production_12,
                                'a_product_no_12':a_product_no_12,
                                'a_work_normal_time_12':a_work_normal_time_12,
                                'a_work_over_time_12':a_work_over_time_12,
                                'a_work_availability_time_12':a_work_availability_time_12, 
                                'a_work_remark_12':a_work_remark_12
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
                                alert(xhr.status);
                                alert(xhr.responseText);
                                alert(throwError);
                                alert(ajaxError);
                        },
                        success:function(res){
                                
                                $("#menu_money_record_list").show(1000).html(res);
                                alert(b_date + ' , 新增完成。');
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("submit 液劑工時紀錄清單 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });
        }

}

function auto_plus_availability_work_time(){
        var a_work_availability_time_1       = $('#a_work_availability_time_1').val();
        var a_work_availability_time_2       = $('#a_work_availability_time_2').val();
        var a_work_availability_time_3       = $('#a_work_availability_time_3').val();
        var a_work_availability_time_4       = $('#a_work_availability_time_4').val();
        var a_work_availability_time_5       = $('#a_work_availability_time_5').val();
        var a_work_availability_time_6       = $('#a_work_availability_time_6').val();
        var a_work_availability_time_7       = $('#a_work_availability_time_7').val();
        var a_work_availability_time_8       = $('#a_work_availability_time_8').val();
        var a_work_availability_time_9       = $('#a_work_availability_time_9').val();
        var a_work_availability_time_10       = $('#a_work_availability_time_10').val();
        var a_work_availability_time_11       = $('#a_work_availability_time_11').val();
        var a_work_availability_time_12       = $('#a_work_availability_time_12').val();

        // auto plus availability work time
        var auto_plus_availability_work_time_val = (Number(a_work_availability_time_1) + Number(a_work_availability_time_2) + Number(a_work_availability_time_3)  + Number(a_work_availability_time_4)  + Number(a_work_availability_time_5)  + Number(a_work_availability_time_6) + 
                                             Number(a_work_availability_time_7) + Number(a_work_availability_time_8) + Number(a_work_availability_time_9)  + Number(a_work_availability_time_10)  + Number(a_work_availability_time_11)  + Number(a_work_availability_time_12)).toFixed(1);
        $('#auto_plus_availability_work_time').show().html(auto_plus_availability_work_time_val);
        
        // auto plus total work time
        //var auto_plus_total_work_time_val = (Number($('#auto_plus_over_work_time').text()) + Number($('#auto_plus_normal_work_time').text()) + Number($('#auto_plus_availability_work_time').text())).toFixed(1)
        var auto_plus_total_work_time_val = (Number($('#auto_plus_over_work_time').text()) + Number($('#auto_plus_normal_work_time').text())).toFixed(1)

        $('#auto_plus_total_work_time').val(auto_plus_total_work_time_val);
}

function auto_plus_over_work_time(){
        var a_work_over_time_1       = $('#a_work_over_time_1').val();
        var a_work_over_time_2       = $('#a_work_over_time_2').val();
        var a_work_over_time_3       = $('#a_work_over_time_3').val();
        var a_work_over_time_4       = $('#a_work_over_time_4').val();
        var a_work_over_time_5       = $('#a_work_over_time_5').val();
        var a_work_over_time_6       = $('#a_work_over_time_6').val();
        var a_work_over_time_7       = $('#a_work_over_time_7').val();
        var a_work_over_time_8       = $('#a_work_over_time_8').val();
        var a_work_over_time_9       = $('#a_work_over_time_9').val();
        var a_work_over_time_10       = $('#a_work_over_time_10').val();
        var a_work_over_time_11       = $('#a_work_over_time_11').val();
        var a_work_over_time_12       = $('#a_work_over_time_12').val();

        // auto plus over work time
        var auto_plus_over_work_time_val = (Number(a_work_over_time_1) + Number(a_work_over_time_2) + Number(a_work_over_time_3)  + Number(a_work_over_time_4)  + Number(a_work_over_time_5)  + Number(a_work_over_time_6) + 
                                             Number(a_work_over_time_7) + Number(a_work_over_time_8) + Number(a_work_over_time_9)  + Number(a_work_over_time_10)  + Number(a_work_over_time_11)  + Number(a_work_over_time_12)).toFixed(1);
        
        $('#auto_plus_over_work_time').show().html(auto_plus_over_work_time_val);


        // auto plus total work time
        //var auto_plus_total_work_time_val = (Number($('#auto_plus_over_work_time').text()) + Number($('#auto_plus_normal_work_time').text()) + Number($('#auto_plus_availability_work_time').text())).toFixed(1)
        var auto_plus_total_work_time_val = (Number($('#auto_plus_over_work_time').text()) + Number($('#auto_plus_normal_work_time').text())).toFixed(1)

        $('#auto_plus_total_work_time').val(auto_plus_total_work_time_val);
}  

function auto_plus_normal_work_time(){
        var a_work_normal_time_1       = $('#a_work_normal_time_1').val();
        var a_work_normal_time_2       = $('#a_work_normal_time_2').val();
        var a_work_normal_time_3       = $('#a_work_normal_time_3').val();
        var a_work_normal_time_4       = $('#a_work_normal_time_4').val();
        var a_work_normal_time_5       = $('#a_work_normal_time_5').val();
        var a_work_normal_time_6       = $('#a_work_normal_time_6').val();
        var a_work_normal_time_7       = $('#a_work_normal_time_7').val();
        var a_work_normal_time_8       = $('#a_work_normal_time_8').val();
        var a_work_normal_time_9       = $('#a_work_normal_time_9').val();
        var a_work_normal_time_10      = $('#a_work_normal_time_10').val();
        var a_work_normal_time_11      = $('#a_work_normal_time_11').val();
        var a_work_normal_time_12      = $('#a_work_normal_time_12').val();

        // auto plus normal work time
        var auto_plus_normal_work_time_val = (Number(a_work_normal_time_1) + Number(a_work_normal_time_2) + Number(a_work_normal_time_3)  + Number(a_work_normal_time_4)  + Number(a_work_normal_time_5)  + Number(a_work_normal_time_6) + 
                                             Number(a_work_normal_time_7) + Number(a_work_normal_time_8) + Number(a_work_normal_time_9)  + Number(a_work_normal_time_10)  + Number(a_work_normal_time_11)  + Number(a_work_normal_time_12)).toFixed(1);

        $('#auto_plus_normal_work_time').show().html(auto_plus_normal_work_time_val);

        // auto plus total work time
        //var auto_plus_total_work_time_val = (Number($('#auto_plus_over_work_time').text()) + Number($('#auto_plus_normal_work_time').text()) + Number($('#auto_plus_availability_work_time').text())).toFixed(1)
        var auto_plus_total_work_time_val = (Number($('#auto_plus_over_work_time').text()) + Number($('#auto_plus_normal_work_time').text())).toFixed(1)

        $('#auto_plus_total_work_time').val(auto_plus_total_work_time_val);

}    

function submit_add_account_form(){
        
        var a_work_no  = $('#a_work_no').val();
        var a_date     = $('#a_date').val();
        var a_name     = $('#a_name').val();
        var a_user     = $('#a_user').val();
        var a_position = $('#a_position').val();
        var a_status   = $('#a_status').val();
        
        var data    = a_date.split('-')
        var r_year  = data[0];
        var r_month = data[1];
        var r_day   = data[2];

        // check 帳號
	if(a_user.length == 0){
	        /// show msg
                alert('帳號不能空白 !!!')
	        exit;
	}
        // check 工號
	if(a_work_no.length == 0){
	        /// show msg
                alert('工號不能空白 !!!')
	        exit;
	}
        // check 姓名 
	if(a_name.length == 0){
	        /// show msg
                alert('姓名不能空白 !!!')
	        exit;
	}
        // check 部門 
	if(a_position.length == 0){
	        /// show msg
                alert('部門不能空白 !!!')
	        exit;
	}
        // check 帳號狀態
	if(a_status.length == 0){
	        /// show msg
                alert('狀態不能空白 !!!')
	        exit;
	}

        //alert(a_work_no + ' / ' +  a_date + ' / ' + a_name + ' / ' + a_position + ' / ' + a_status)
        //exit();

        $.ajax({
                type:"POST",
                url:"/submit_add_account_form",
                data:{
                        'a_user':a_user,
                        'a_name':a_name,
                        'a_date':a_date,
                        'a_work_no':a_work_no,
                        'a_position':a_position,
                        'a_status':a_status
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
                        
                        // clean value
                        $('#a_work_no').val('');
                        $('#a_date').val('');
                        $('#a_name').val('');
                        $('#a_user').val('');
                        $('#a_position').val('');
                        $('#a_status').val('');
                        
                        // reload account list
                        reload_menu_account_list();
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        //$('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });     
}

function logout2(){
        
        // scroll to top 
        jQuery("html,body").animate({scrollTop:0},1000);

        $.ajax({
                type:"GET",
                url:"/logout2",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError);
                },
                success:function(res){
                        alert("超過10分鐘沒任何度動作 , 系統已將您已自動登出 !");
                        window.location.href="/login"
                        //$("#add_account_form").show(1000).html(res);
                },
                beforeSend:function(){
                        $('#status').html("now logout...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function add_account_form(){
        
        // scroll to top 
        jQuery("html,body").animate({scrollTop:0},1000);

        $.ajax({
                type:"POST",
                url:"/load_account_form",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError);
                },
                success:function(res){
        	       	$("#add_account_form").show(1000).html(res);
                },
                beforeSend:function(){
                        $('#status').html("loading 新增帳號表 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function reload_menu_account_list(){
        $.ajax({
                type:"GET",
                url:"/reload_menu_account_list",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
                        
                        //location.reload(true);
                        $("#otsuka_account_list").show(1000).html(res);  
                },
                beforeSend:function(){
                        $('#status').html("loading 帳號清單 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });                
}

function load_menu_money_record_by_kind(val){
        var kind = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_money_record_by_kind",
                        data:{
                                'kind':kind
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError);
                        },
                        success:function(res){
                                
                                $("#menu_money_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + kind + " 種類記帳本清單 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_money_record_by_day(val){
        var day = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_money_record_by_day",
                        data:{
                                'day':day
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_money_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + day + " 日記帳本清單 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_money_record_by_month(val){
        var month = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_money_record_by_month",
                        data:{
                                'month':month
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_money_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + month + " 月記帳本清單 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_car_record_by_day(val){
        var day = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_car_record_by_day",
                        data:{
                                'day':day
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_car_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + day + " 日用車記錄 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_car_record_by_month(val){
        var month = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_car_record_by_month",
                        data:{
                                'month':month
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_car_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + month + " 月用車記錄 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_car_record_by_year(val){
        var year = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_car_record_by_year",
                        data:{
                                'year':year
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_car_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + year + " 年用車記錄 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_money_record_by_year(val){
        var year = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_money_record_by_year",
                        data:{
                                'year':year
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_money_record_list").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + year + " 年記帳本清單 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_calendar_record_by_month(val){
        var month = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_calendar_record_by_month",
                        data:{
                                'month':month
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_calendar_record_content").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + month + " 月工作日誌 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function load_menu_work_record_by_kind(val){
        var kind = val;
        
                $.ajax({
                        type:"POST",
                        url:"/load_menu_work_record_by_kind",
                        data:{
                                'kind':kind
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                $("#menu_work_record_content").show(1000).html(res);
                                
                                // scroll page bottom to page top
                                goto_top();
                                
                                //location.reload(true);
                        },
                        beforeSend:function(){
                                $('#status').html("loading " + kind + " 工作記錄 ...").css({'color':'blue'});
                        },
                        complete:function(){
                                $('#status').css({'color':'white'});
                        }
                });

}

function del_menu_car_record(val){
        
        var del_no = val;
        
        var check_del = prompt("刪除 No." + del_no + " , 確定刪除 , 再按一次 y ");
        
	if(check_del == 'y'){	
                $.ajax({
                        type:"POST",
                        url:"/del_menu_car_record",
                        data:{
                                'del_no':del_no
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                location.reload(true);
                        },
                        beforeSend:function(){
                                $('#menu_money_record_list').show(1000);
                        },
                        complete:function(){
                        }
                });
	}else{
                exit();
        }
}

function del_menu_money_record(val){
        
        var del_no = val;
        
        var check_del = prompt("刪除 No." + del_no + " , 確定刪除 , 再按一次 y ");
        
	if(check_del == 'y'){	
                $.ajax({
                        type:"POST",
                        url:"/del_menu_money_record",
                        data:{
                                'del_no':del_no
                        },
                        datatype:"html",
                                error:function(xhr , ajaxError , throwError){
         	                alert(xhr.status);
               	                alert(xhr.responseText);
	                        alert(throwError);
                                alert(ajaxError)
                        },
                        success:function(res){
                                
                                //$('#menu_money_record_list').show(1000).html(res);
                                
                                // reload menu money record
                                reload_menu_money_record();
                        },
                        beforeSend:function(){
                                //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                                $('#menu_money_record_list').show(1000);
                        },
                        complete:function(){
                                //$('#show_msg').hide();
                        }
                });
	}else{
                exit();
        }
}

function goto_top(){
        
        // scroll page bottom to page top
        jQuery("html,body").animate({scrollTop:0},1000);
        $('#goto_top').css({'cursor':'pointer'});

}

function submit_alter_calendar_record_form(){
        
        var no      = $('#no').val();
        var r_time  = $('#record_time').val();
        var title   = $('#title').val();
        var content = CKEDITOR.instances.content.getData();

        $.ajax({
                type:"POST",
                url:"/submit_alter_calendar_record",
                data:{
                        'no':no,
                        'r_time':r_time,
                        'title':title,
                        'content':content
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
			
                        alert(title + '  , 修改完成。');
                        location.reload(true);
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#detail_calendar_record_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });
}

function submit_alter_work_record_form(){
        
        var no      = $('#no').val();
        var r_time  = $('#record_time').val();
        var kind    = $('#kind').val();
        var title   = $('#title').val();
        var content = CKEDITOR.instances.content.getData();

        $.ajax({
                type:"POST",
                url:"/submit_alter_work_record",
                data:{
                        'no':no,
                        'r_time':r_time,
                        'kind':kind,
                        'title':title,
                        'content':content
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
			//$("#content").css({'border':'#cccccc 1px solid'});
        	       	//$("#detail_work_record_content").show(1000).html(res);
                        
                        alert(kind + ' - ' + title + '  , 修改完成。');

                        location.reload(true);
                        // load alter  detail work record
                        //detail_work_record(no);

                        // reload menu work record
                        //reload_menu_work_record();
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#detail_work_record_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });
        

}

function detail_calendar_record(val){
        
        var no = val;
        
        // scroll to top 
        jQuery("html,body").animate({scrollTop:0},1000);

        $.ajax({
                type:"POST",
                url:"/detail_calendar_record",
                data:{
                        'no':no
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
			//$("#content").css({'border':'#cccccc 1px solid'});
                        
        	       	$("#detail_calendar_record_content").show(1000).html(res);
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#detail_calendar_record_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });
}

function detail_work_record(val){

        var no = val;
        
        // scroll to top 
        jQuery("html,body").animate({scrollTop:0},1000);

        $.ajax({
                type:"POST",
                url:"/detail_work_record",
                data:{
                        'no':no
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
			//$("#content").css({'border':'#cccccc 1px solid'});
        	       	$("#detail_work_record_content").show(1000).html(res);
                },
                beforeSend:function(){
                        $('#status').html("loading " + no + " 工作記錄 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });
}

function submit_add_work_record_form(){
        
        var user = $('#user').val();
        var date = $('#date').val();
        var kind = $('#kind').val();
        var title = $('#title').val();
        var content = CKEDITOR.instances.content.getData();

        //alert(user + ' / ' + date + ' / ' + kind + ' / ' + money + ' / ' + content + ' / ' + record_year + ' / ' + record_month + ' / ' + record_day)
        
        // check kind 
	if(kind.length == 0){
	        /// show msg
                $("#kind").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 工作記錄種類不能空白 !!!");
	        exit;
	}
        // check title
	if(title.length == 0){
	        /// show msg
                $("#title").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 工作記錄標體不能空白 !!!");
	        exit;
	}
        // check content 
	if(content.length == 0){
	        /// show msg
                $("#content").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 工作記錄內容不能空白 !!!");
	        exit;
	}

        $.ajax({
                type:"POST",
                url:"/submit_add_work_record_form",
                data:{
                        'user':user,
                        'date':date,
                        'kind':kind,
                        'title':title,
                        'content':content
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
                        //console.log(res.validate);
			//$("#content").css({'border':'#cccccc 1px solid'});
        	       	//$("#add_content").show(1000).html(res);
                        
                        // clean show msg value
                        $("#show_msg").val('');
                        
                        // reload menu work record
                        reload_menu_work_record();
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        

}

function submit_add_car_record_form(){
        
        var user = $('#user').val();
        var date = $('#date').val();
        var kind = $('#kind').val();
        var go_out_km = $('#go_out_km').val();
        var go_home_km = $('#go_home_km').val();
        var total_used_km = go_home_km - go_out_km;
        var destination = $('#destination').val();
        var data = date.split('-');
        var r_year = data[0];
        var r_month = data[0]+'-'+data[1];
        var r_day = data[2];

        
        // check kind 
	if(kind.length == 0){
	        /// show msg
                $("#kind").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 用車種類不能空白 !!!");
	        exit;
	}
        // check go_home_km
	if(go_home_km.length == 0){
	        /// show msg
                $("#go_home_km").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 入庫里程不能空白 !!!");
	        exit;
	}
        // check destination
	if(destination.length == 0){
	        /// show msg
                $("#destination").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 用車記錄內容不能空白 !!!");
	        exit;
	}



        $.ajax({
                type:"POST",
                url:"/submit_add_car_record_form",
                data:{
                        'user':user,
                        'date':date,
                        'kind':kind,
                        'go_out_km':go_out_km,
                        'go_home_km':go_home_km,
                        'total_used_km':total_used_km,
                        'destination':destination,
                        'r_year':r_year,
                        'r_month':r_month,
                        'r_day':r_day
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
                        
                        //alert('ok');
                        //$("#add_content").show(1000).html(res);
                        
                        // clean show msg value
                        //$("#show_msg").val('');
                        
                        // reload menu money record
                        //reload_menu_money_record();
                        location.reload(true);
                        
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        

}

function submit_add_money_record_form(){
        
        var user = $('#user').val();
        var date = $('#date').val();
        var kind = $('#kind').val();
        var money = $('#money').val();
        var content = $('#content').val();
        var data1 = date.split(' ')
        var data2 = data1[0].split('-')
        var record_year = data2[0];
        var record_month = data2[1];
        var record_day = data2[2];

        //alert(user + ' / ' + date + ' / ' + kind + ' / ' + money + ' / ' + content + ' / ' + record_year + ' / ' + record_month + ' / ' + record_day)
        
        // check kind 
	if(kind.length == 0){
	        /// show msg
                $("#kind").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 記帳表種類不能空白 !!!");
	        exit;
	}
        // check money
	if(money.length == 0){
	        /// show msg
                $("#money").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 記帳表花費不能空白 !!!");
	        exit;
	}
        // check content 
	if(content.length == 0){
	        /// show msg
                $("#content").css({'border-bottom-color':'red'});
		$("#show_msg").css({'color':'red'}).html("<i class='bi bi-x-circle-fill'></i> 記帳表內容不能空白 !!!");
	        exit;
	}

        $.ajax({
                type:"POST",
                url:"/submit_add_money_record_form",
                data:{
                        'user':user,
                        'date':date,
                        'kind':kind,
                        'money':money,
                        'content':content,
                        'record_year':record_year,
                        'record_month':record_month,
                        'record_day':record_day
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){
                        
                        $("#add_content").show(1000).html(res);
                        
                        // clean show msg value
                        $("#show_msg").val('');
                        
                        // reload menu money record
                        //reload_menu_money_record();

                        location.reload(true);
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        

}

function reload_menu_money_record_by_day(){
        $.ajax({
                type:"GET",
                url:"/reload_menu_money_record_by_day",
                //url:"/menu_calendar_record",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
                        
                        location.reload(true);
                        $("#money_record_by_day").show(1000).html(res);  
                },
                beforeSend:function(){
                        $('#status').html("loading 記帳本 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });                
}

function reload_menu_car_record(){
        $.ajax({
                type:"GET",
                url:"/reload_menu_car_record",
                //url:"/menu_calendar_record",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
                        
                        location.reload(true);
                        $("#menu_car_record_list").show(1000).html(res);  
                },
                beforeSend:function(){
                        $('#status').html("loading 用車記錄 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });                
}

function reload_menu_money_record(){
        $.ajax({
                type:"GET",
                url:"/reload_menu_money_record",
                //url:"/menu_calendar_record",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
                        
                        location.reload(true);
                        $("#menu_money_record_list").show(1000).html(res);  
                },
                beforeSend:function(){
                        $('#status').html("loading 記帳本 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });                
}

function reload_menu_calendar_record(){
        $.ajax({
                type:"GET",
                url:"/reload_menu_calendar_record",
                //url:"/menu_calendar_record",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
                        
                        location.reload(true);
                        $("#main_content").show(1000).html(res);  
                },
                beforeSend:function(){
                        $('#status').html("loading 工作日誌 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });                
}

function reload_menu_work_record(){
        $.ajax({
                type:"GET",
                url:"/reload_menu_work_record",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
                        
                        $("#menu_work_record_content").html(res);  
                        location.reload(true);
                },
                beforeSend:function(){
                        $('#status').html("loading 工作記錄 ...").css({'color':'blue'});
                },
                complete:function(){
                        $('#status').css({'color':'white'});
                }
        });                
}

function select_car_record_kind(val){
        var data = val;
        $('#kind').val(data);

        // select go out km 
        $.ajax({
                type:"POST",
                url:"/select_car_record_by_go_out_km",
                data:{
                        'kind':val
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){

                        $('#go_out_km').val(res);
                },
                beforeSend:function(){
                        $('#detail_calendar_record_content').show(1000);
                },
                complete:function(){
                        
                }
        });

}

function select_money_record_kind(val){
        var data = val;
        $('#kind').val(data);
}

function del_alter_calendar_record_form(){
        
        var no = $('#no').val();

        var check_del = prompt("刪除 No." + no + " ， 確定請再按一次 y : ");
	if(check_del != 'y'){	
                exit();
	}else{

                $.ajax({
                type:"POST",
                url:"/del_alter_calendar_record_form",
                data:{
                        'no':no
                },
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                      alert(ajaxError)
                },
                success:function(res){

                        $('#detail_calendar_record_content').hide(1000);
                        
                        // reload menu calendar record
                        reload_menu_calendar_record();
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#detail_calendar_record_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
                });
        }
}

function cancel_add_work_record_form(){
        $('#kind').val('');
        $('#content').val('');
        $('#title').val('');

        $("#add_work_form").hide(1000);
        location.reload(true);
}

function cancel_add_money_record_form(){
        $('#kind').val('');
        $('#content').val('');
        $('#money').val('');

        $("#add_money_form").hide(1000);
        location.reload(true);
}

function add_calendar_record(){
        
        // scroll page bottom to page top 
        jQuery("html,body").animate({scrollTop:0},1000);

        $.ajax({
                type:"GET",
                url:"/add_calendar_record_form",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
			
                        //$("#content").css({'border':'#cccccc 1px solid'});
        	       	$("#add_content").show(1000).html(res);

                        // hide alter work record form content
                        $('#detail_work_record_content').hide(1000);
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        
}

function add_work_record(){
        
        // scroll to top 
        jQuery("html,body").animate({scrollTop:0},1000);

        $.ajax({
                type:"GET",
                url:"/add_work_record_form",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
			
                        //$("#content").css({'border':'#cccccc 1px solid'});
        	       	$("#add_content").show(1000).html(res);

                        // hide alter work record form content
                        $('#detail_work_record_content').hide(1000);
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        
}

function add_car_record(){
        $.ajax({
                type:"GET",
                url:"/add_car_record_form",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
			//$("#content").css({'border':'#cccccc 1px solid'});
        	       	$("#add_content").show(1000).html(res);
                               
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        
}

function add_website_record(){
        $.ajax({
                type:"GET",
                url:"/add_website_record_form",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
			//$("#content").css({'border':'#cccccc 1px solid'});
        	       	$("#add_content").show(1000).html(res);
                               
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        
}

function add_money_record(){
        $.ajax({
                type:"GET",
                url:"/add_money_record_form",
                data:{},
                datatype:"html",
                error:function(xhr , ajaxError , throwError){
         	      alert(xhr.status);
               	      alert(xhr.responseText);
	              alert(throwError);
                },
                success:function(res){
			//$("#content").css({'border':'#cccccc 1px solid'});
        	       	$("#add_content").show(1000).html(res);
                               
                },
                beforeSend:function(){
                        //$('#show_msg').html('載入操作紀錄管理中...').css({'color':'blue'}).show();
                        $('#add_content').show(1000);
                },
                complete:function(){
                        //$('#show_msg').hide();
                }
        });        
}
