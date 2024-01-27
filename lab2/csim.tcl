catch {::common::set_param -quiet hls.xocc.mode csynth};
# 
# HLS run script generated by v++ compiler
# 

open_project CnnKernel
set_top CnnKernel
# v++ -g, -D, -I, --advanced.prop kernel.CnnKernel.kernel_flags
add_files ".merlin_prj/run/implement/exec/hls/__merlinkernel_CnnKernel.cpp" -cflags " -O3 -D XILINX -I /home/ubuntu/cs-259-w24/lab1/.merlin_prj/run/implement/exec/hls"
add_files -tb ".merlin_prj/run/implement/exec/hls/__merlinkernel_CnnKernel_tb.cpp"
open_solution -flow_target vitis solution
set_part xcu200-fsgd2104-2-e
# v++ --hls.clock or --kernel_frequency
create_clock -period 250MHz -name default
csim_design
close_project
puts "HLS completed successfully"
exit
