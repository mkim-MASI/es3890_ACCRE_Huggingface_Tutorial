# es3890_ACCRE_Huggingface_Tutorial
Tutorial for using the ACCRE GPU resources to run Huggingface models 

Instructor: Michael Kim

- To Start
- Setting up
    - Job Submission
    - Interactive Session
    - Creating and Activating Environment
- Part 1: Large Language Model (LLM) for Text Generation
- Part 2: Stable Diffusion Model for Image Generation
- Part 3: Vision Language Model
- Part 4: Additional Models

---

## To Start

Start the Desktop interactive session, since it takes a while to start up!

We will be using the Advanced Computing Center for Research and Education (ACCRE) at Vanderbilt University for the computational resources to run this tutorial.
As a refresher, please follow the instructions in this section to make sure that you can connect and have access to the ACCRE resources for the class:

1. Make sure that you can login to the ACCRE dashboard with your VUnet ID and password here: https://viz.accre.vu/pun/sys/dashboard
2. Click on the "Files" dropdown menu and click "Home Directory"
3. Click on the button near the top that says "Open in Terminal". It should open a new window with a terminal being run on an ACCRE login node.
4. Run the command `slurm_groups`. You should see that you are a member of both `es3890_acc` and `es3890_iacc`
5. Run the command `salloc --account=es3890_acc --partition=batch_gpu --gres=gpu:nvidia_rtx_a6000:1 --time=00:05:00`. It should allocate a GPU node to you in a few seconds.
6. Run the command `nvidia-smi`. You should see an output that looks similar to:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A4000               On  |   00000000:86:00.0 Off |                  Off |
| 41%   32C    P8             15W /  140W |       2MiB /  16376MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

If you see something similar, then you have successfully connected to ACCRE GPU resources.

---

## Setting up

Some people may feel more or less comfortable using ACCRE resources. For those more comfortable submitting batch jobs to the cluster,
feel free to run all Python code below by submitting ACCRE jobs. However, if you have less familiarity (or prefer interactive environments instead like me),
all the code below can be run interactively as well.

---

#### Job Submission

If you have a block of Python code:

```
#############
# CODE BLOCK
#############
```

You can submit this code as a job to run on the ACCRE cluster. Simply create a new file, called `job.slurm` that is strcutured the following way:

```
#SBATCH --mail-user=<EMAIL>             # For sending email when job is complete
#SBATCH --mail-type=FAIL,END            # Situations when you get notified by email
#SBATCH --account=es3890_acc            # Project account name
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Specifying the type of GPU: --gres=gpu:<gpu-type>:<gpu-num>
#SBATCH --partition=batch_gpu           # Partition to use for the job submission
#SBATCH --nodes=1                       # Number of nodes to request (currently one node)
#SBATCH --mem-per-cpu=16G               # Memory per CPU core (currently 16GB)
#SBATCH --time=00:05:00                 # Maximum runtime (D-HH:MM:SS or HH:MM:SS)

#activate your project environment here
#conda activate my_env
#source my_env/bin/activate

#############
# CODE BLOCK
#############

```

You can then submit the job by running: `sbatch job.slurm`

---

#### Interactive Session

There are two ways to start an interactive session for an ACCRE GPU node. One is to use the ACCRE visualization portal following the instructions below:

1. On the ACCRE dashboard/visualization portal website, click on the "Interactive Apps" dropdown menu and click "ACCRE Desktop"
2. Fill out the number of hours you would like to have allocated to you (1 hour should be enough for the tutorial)
3. Fill out the ACCRE slurm account, `es3890_acc`
4. In the dropdown for "Partition", select "batch_gpu"
5. For the "GPU Type" dropdown, select "NVIDIA RTX A6000"
6. For "Number of GPUs", enter "1"
7. Click "Launch" at the bottom of the page

This should take a few minutes before the resources are allocated to you. 

If you prefer the simpler way, you can run the following from the terminal to also start an interactive session directly from the command line:
```
salloc --account=es3890_acc --partition=batch_gpu --gres=gpu:nvidia_rtx_a6000:1 --time=02:00:00 --mem=48G
```

---

Regardless of whichever method you choose, everyone must (should) set up a Python environment to run for this tutorial:

#### Creating and Activating Environment

1. Load a more recent version of Python. This tutorial works with version `3.12.4`, which can be loaded running `module load python/3.12.4`.
2. Create a virtual environment by running `python3 -m venv <ENVIRONMENT_NAME>`, where `<ENVIRONMENT_NAME>` is the name you wish to call the environment.
3. Activate the environment by running `source <ENVIRONMENT_NAME>/bin/activate`

Any additional libraries/codebases you download while the environment is activated will be stored here.
Environments are beneficial for keeping dependencies of different projects separated.

Anytime you wish to deativate the environment, simply run `deactivate`.


## Part 1: Large Language Model (LLM) for Text Generation

The model and code in this tutorial come from the following huggingface repo: https://huggingface.co/Qwen/Qwen3-0.6B.
This model is for a text generation model that responds to a given prompt.

To begin, make sure your environment is activated.
We must then install the necessary dependencies in order to run the code for the LLM by running:

```
pip install torch transformers accelerate
```

Once these libraries finish installing, create a Python file called `test_huggingface1.py` with the following code:

```
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-0.6B"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
# prepare the model input
prompt = "Please tell me about Vanderbilt University. Where is it located and what is it like?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

You can then simply run the Python code by calling `python3 test_huggingface1.py`.
You should see output lines that are in response to the prompt "Please tell me about Vanderbilt University. Where is it located and what is it like?"

Note that you can play around with this, changing the line `prompt = ...` to see what different outputs you can get.

---

## Part 2: Stable Diffusion Model for Image Generation

The model and code in this tutorial comes from here: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0.
This model is for an image generation model, where the image is generated based on the given prompt.
This is, in my opinion, the most fun and my personal favorite of the three parts (and I know Dr. Landman's favorite as well!)

To begin, make sure your environment is activated.
We must then install the necessary dependencies in order to run the code for the image generation model by running:

```
pip install diffusers
```

Once this library finishes installing, create a new folder/directory by running `mkdir part2_outputs`, where `part2_outputs` is the name of the directory.
Then, create a Python file called `test_huggingface2.py` with the following code:

```
import torch
import os
from diffusers import DiffusionPipeline
output_dir="~/part2_outputs/"
#load the base model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
pipe.to("cuda")
# Generate an image with the base model
prompt = "Cat sitting on the beach looking at birds."
image = pipe(prompt=prompt).images[0]
# Save the generated image
output_path = os.path.join(output_dir, "testimg.png")
image.save(output_path)
```

If you are using an interactive session from the terminal directly, then you can look at the resulting image on the ACCRE dashboard.
Navigate to the "Files" dropdown menu and click on the `part2_outputs` folder.
Click on the newly created `testimg.png` to see the image that should match the prompt "Cat sitting on the beach looking at birds".

Play around with this by changing the prompt to see what kinds of images you can create!

---

## Part 3: Vision Language Model

The model and code in this tutorial comes from here: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0.
This model is for a vision language model, where, given an image and a prompt, generates a text response and a bounding box(es) for the image. 
(The model can also accept a bounding box as well.)

To begin, make sure your environment is activated.
We have installed all the necessary dependencies from the previous parts of this tutorial.

Create a new folder/directory by running `mkdir part3_outputs`, where `part3_outputs` is the name of the directory.
Then, create a Python file called `test_huggingface3.py` with the following code:

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': 'Generate the caption in English with grounding:'},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)

image = tokenizer.draw_bbox_on_latest_picture(response)
if image:
  image.save('~/part3_outputs/testimg.jpg')
else:
  print("no box")
```

You should see a text response that corresponds to the original image here: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg.
The response will also be linked to the newly generated image from the model.
If you are using an interactive session from the terminal directly, then you can look at the resulting image on the ACCRE dashboard.
Navigate to the "Files" dropdown menu and click on the `part3_outputs` folder.
Click on the newly created `testimg.jpg` to see the image that was generated.
It should be the original image with an added bounding box(es) that corresponds to the prompt.

Play around with this by changing the prompt and the input image to see what changes in the outputs.

---

## Part 4: Additional Models

There are many more models available than these ones demonstrated here. You can search for available ones on the Huggingface website: https://huggingface.co/models.
Some will likely be easier than others to follow/get running. Here are some examples that may be useful in some of your projects:

For ranking similarities between sets of texts and images: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B

For character recognition in images: https://huggingface.co/PaddlePaddle/PaddleOCR-VL

Think about how some of these models, including the ones in this tutorial, could potentially be used in some of your projects!


