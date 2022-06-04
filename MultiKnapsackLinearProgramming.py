from pulp import *

def run():
    MAX_N = 10000 # a very large number
    # Input
    total_hosts = 3

    task0_duration = 2
    task1_duration = 3
    task2_duration = 5
    task3_duration = 7
    task4_duration = 9
    total_time_to_execute_tasks_sequential = task0_duration + task1_duration + task2_duration + task3_duration + task4_duration

    # Aim is to minimize the objective
    prob = LpProblem("myProblem", LpMinimize)

    # define variables
    ideal_time_for_each_host = LpVariable("ideal_time_for_each_host", 0, MAX_N, cat="Continuous")
    time_tasks_took_on_host0 = LpVariable('time_tasks_took_on_host0', 0, MAX_N,  cat="Integer")
    time_tasks_took_on_host1 = LpVariable('time_tasks_took_on_host1', 0, MAX_N,  cat="Integer")
    time_tasks_took_on_host2 = LpVariable('time_tasks_took_on_host2', 0, MAX_N,  cat="Integer")
    # Booleans 
    if_host0_ran_task0 = LpVariable('if_host0_ran_task0', 0, 1, cat="Binary")
    if_host0_ran_task1 = LpVariable('if_host0_ran_task1', 0, 1, cat="Binary")
    if_host0_ran_task2 = LpVariable('if_host0_ran_task2', 0, 1, cat="Binary")
    if_host0_ran_task3 = LpVariable('if_host0_ran_task3', 0, 1, cat="Binary")
    if_host0_ran_task4 = LpVariable('if_host0_ran_task4', 0, 1, cat="Binary")
    #
    if_host1_ran_task0 = LpVariable('if_host1_ran_task0', 0, 1, cat="Binary")
    if_host1_ran_task1 = LpVariable('if_host1_ran_task1', 0, 1, cat="Binary")
    if_host1_ran_task2 = LpVariable('if_host1_ran_task2', 0, 1, cat="Binary")
    if_host1_ran_task3 = LpVariable('if_host1_ran_task3', 0, 1, cat="Binary")
    if_host1_ran_task4 = LpVariable('if_host1_ran_task4', 0, 1, cat="Binary")
    #
    if_host2_ran_task0 = LpVariable('if_host2_ran_task0', 0, 1, cat="Binary")
    if_host2_ran_task1 = LpVariable('if_host2_ran_task1', 0, 1, cat="Binary")
    if_host2_ran_task2 = LpVariable('if_host2_ran_task2', 0, 1, cat="Binary")
    if_host2_ran_task3 = LpVariable('if_host2_ran_task3', 0, 1, cat="Binary")
    if_host2_ran_task4 = LpVariable('if_host2_ran_task4', 0, 1, cat="Binary")

    # Variable to hold diffs
    time_tasks_took_on_host0_minus_ideal = LpVariable('time_tasks_took_on_host0_minus_ideal',  -MAX_N, MAX_N, cat="Continuous")
    time_tasks_took_on_host1_minus_ideal = LpVariable('time_tasks_took_on_host1_minus_ideal',  -MAX_N, MAX_N, cat="Continuous")
    time_tasks_took_on_host2_minus_ideal = LpVariable('time_tasks_took_on_host2_minus_ideal',  -MAX_N, MAX_N, cat="Continuous")

    abs_time_tasks_took_on_host0_minus_ideal = LpVariable('abs_time_tasks_took_on_host0_minus_ideal', 0, MAX_N,cat="Continuous")
    abs_time_tasks_took_on_host1_minus_ideal = LpVariable('abs_time_tasks_took_on_host1_minus_ideal', 0, MAX_N,cat="Continuous")
    abs_time_tasks_took_on_host2_minus_ideal = LpVariable('abs_time_tasks_took_on_host2_minus_ideal', 0, MAX_N,cat="Continuous")

    # Objective
    prob += (abs_time_tasks_took_on_host0_minus_ideal + abs_time_tasks_took_on_host1_minus_ideal + abs_time_tasks_took_on_host2_minus_ideal), "objective keep running time on each host close to average"
    # ideal expected time for all tasks for each host is (total/3)
    prob += (ideal_time_for_each_host * 3 == total_time_to_execute_tasks_sequential), "c_deal_time_for_each_host"
    # each task can only run on max one host
    prob += (if_host0_ran_task0 + if_host1_ran_task0 + if_host2_ran_task0 == 1), "task0_should_run_on_max_1_host"
    prob += (if_host0_ran_task1 + if_host1_ran_task1 + if_host2_ran_task1 == 1), "task1_should_run_on_max_1_host"
    prob += (if_host0_ran_task2 + if_host1_ran_task2 + if_host2_ran_task2 == 1), "task2_should_run_on_max_1_host"
    prob += (if_host0_ran_task3 + if_host1_ran_task3 + if_host2_ran_task3 == 1), "task3_should_run_on_max_1_host"
    prob += (if_host0_ran_task4 + if_host1_ran_task4 + if_host2_ran_task4 == 1), "task4_should_run_on_max_1_host"
    # Sum of time to run all tasks on each host equals total time taken to run tasks sequentially
    prob += (time_tasks_took_on_host0 + time_tasks_took_on_host1 + time_tasks_took_on_host2 == total_time_to_execute_tasks_sequential), "total_time_to_execute_tasks_sequential_equals_total_time_of_each_host"
    # Atleast 1 task shold be run on each host
    prob += (if_host0_ran_task0 + if_host0_ran_task1 + if_host0_ran_task2 + if_host0_ran_task3 + if_host0_ran_task4 >= 1), "host0_should_not_be_idle"
    prob += (if_host1_ran_task0 + if_host1_ran_task1 + if_host1_ran_task2 + if_host1_ran_task3 + if_host1_ran_task4 >= 1), "host1_should_not_be_idle"
    prob += (if_host2_ran_task0 + if_host2_ran_task1 + if_host2_ran_task2 + if_host2_ran_task3 + if_host2_ran_task4 >= 1), "host2_should_not_be_idle"
    # Time taken to run tasks on each host
    prob += (time_tasks_took_on_host0 == if_host0_ran_task0*task0_duration + if_host0_ran_task1*task1_duration + if_host0_ran_task2*task2_duration + if_host0_ran_task3*task3_duration + if_host0_ran_task4*task4_duration), "c_time_tasks_took_on_host0"
    prob += (time_tasks_took_on_host1 == if_host1_ran_task0*task0_duration + if_host1_ran_task1*task1_duration + if_host1_ran_task2*task2_duration + if_host1_ran_task3*task3_duration + if_host1_ran_task4*task4_duration), "c_time_tasks_took_on_host1"
    prob += (time_tasks_took_on_host2 == if_host2_ran_task0*task0_duration + if_host2_ran_task1*task1_duration + if_host2_ran_task2*task2_duration + if_host2_ran_task3*task3_duration + if_host2_ran_task4*task4_duration), "c_time_tasks_took_on_host2"

    # calculate deviation from average
    prob += (time_tasks_took_on_host0_minus_ideal == time_tasks_took_on_host0 - ideal_time_for_each_host), "c_time_tasks_took_on_host0_minus_ideal"
    prob += (time_tasks_took_on_host1_minus_ideal == time_tasks_took_on_host1 - ideal_time_for_each_host), "c_time_tasks_took_on_host1_minus_ideal"
    prob += (time_tasks_took_on_host2_minus_ideal == time_tasks_took_on_host2 - ideal_time_for_each_host), "c_time_tasks_took_on_host2_minus_ideal"
    # because we dont have abs function
    prob += (abs_time_tasks_took_on_host0_minus_ideal >= time_tasks_took_on_host0_minus_ideal), "c_abs_time_tasks_took_on_host0_minus_ideal_pos"
    prob += (abs_time_tasks_took_on_host0_minus_ideal >= -time_tasks_took_on_host0_minus_ideal), "c_abs_time_tasks_took_on_host0_minus_ideal_neg"

    prob += (abs_time_tasks_took_on_host1_minus_ideal >= time_tasks_took_on_host1_minus_ideal), "c_abs_time_tasks_took_on_host1_minus_ideal_pos"
    prob += (abs_time_tasks_took_on_host1_minus_ideal >= -time_tasks_took_on_host1_minus_ideal), "c_abs_time_tasks_took_on_host1_minus_ideal_neg"

    prob += (abs_time_tasks_took_on_host2_minus_ideal >= time_tasks_took_on_host2_minus_ideal), "c_abs_time_tasks_took_on_host2_minus_ideal_pos"
    prob += (abs_time_tasks_took_on_host2_minus_ideal >= -time_tasks_took_on_host2_minus_ideal), "c_abs_time_tasks_took_on_host2_minus_ideal_neg"



    # Run Solver
    status = prob.solve()
    print("====================================================")
    print(LpStatus[status])
    print("Ideal time for each host should be (total time for all tasks/ number of hosts) : "+str(value(ideal_time_for_each_host)))
    print("time_tasks_took_on_host0 : "+str(value(time_tasks_took_on_host0)))
    print("time_tasks_took_on_host1 : "+str(value(time_tasks_took_on_host1)))
    print("time_tasks_took_on_host2 : "+str(value(time_tasks_took_on_host2)))

    print("Minimized obj Deviation from ideal average time: "+str(value(abs_time_tasks_took_on_host0_minus_ideal + abs_time_tasks_took_on_host1_minus_ideal + abs_time_tasks_took_on_host2_minus_ideal)))
    print("=====================================================")
    ans = ""
    if value(if_host0_ran_task0)==1:
        ans += "task0,"
    if value(if_host0_ran_task1)==1:
        ans  += "task1,"
    if value(if_host0_ran_task2)==1:
        ans  += "task2,"
    if value(if_host0_ran_task3)==1:
        ans  += "task3,"
    if value(if_host0_ran_task4)==1:
        ans +=  "task4"
    print("Tasks Running on host0: "+ans)

    ans = ""
    if value(if_host1_ran_task0)==1:
        ans += "task0,"
    if value(if_host1_ran_task1)==1:
        ans  += "task1,"
    if value(if_host1_ran_task2)==1:
        ans  += "task2,"
    if value(if_host1_ran_task3)==1:
        ans  += "task3,"
    if value(if_host1_ran_task4)==1:
        ans +=  "task4"
    print("Tasks Running on host1: "+ans)

    ans = ""
    if value(if_host2_ran_task0)==1:
        ans += "task0,"
    if value(if_host2_ran_task1)==1:
        ans  += "task1,"
    if value(if_host2_ran_task2)==1:
        ans  += "task2,"
    if value(if_host2_ran_task3)==1:
        ans  += "task3,"
    if value(if_host2_ran_task4)==1:
        ans +=  "task4"
    print("Tasks Running on host2: "+ans)



if __name__ == '__main__':
    run()
