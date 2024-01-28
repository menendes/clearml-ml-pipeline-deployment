from clearml.automation import PipelineController

project_name = "ihk-mlops"
def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name="Pipeline demo", project=project_name, version="0.0.1", add_pipeline_tags=False
)

pipe.add_parameter("dataset_name", "iris_dataset", "dataset",)
pipe.add_parameter("train_test_dataset_name", "train_test_data", "train and test data",)

pipe.set_default_execution_queue("ihk")

pipe.add_step(
    name="stage_data",
    base_task_project=project_name,
    base_task_name="step_1",
    parameter_override={"General/dataset_name": "${pipeline.dataset_name}"},
)

pipe.add_step(
    name="stage_process",
    parents=["stage_data"],
    base_task_project=project_name,
    base_task_name="step_2",
    parameter_override={
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/test_size": 0.25,
    },
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
)
pipe.add_step(
    name="stage_train",
    parents=["stage_process"],
    base_task_project=project_name,
    base_task_name="step_3",
    parameter_override={"General/dataset_name": "${pipeline.train_test_dataset_name}"},
)

# for debugging purposes use local jobs
# pipe.start_locally()
print("before start")
# Starting the pipeline (in the background)
pipe.start(queue="ihk")

print("done")
