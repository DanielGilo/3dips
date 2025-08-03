import subprocess

scene_1 = {"path": "/home/danielgilo/3dips/seva/weka/home-jensen/reve/datasets/reconfusion_export/co3d-viewcrafter/car",
           "prompt": "a high-quality, detailed, and professional image of a light-blue car",
           "editing_prompts": ["Make the car red",
                               "Make it night",
                               "Make it in the style of Van Gogh",
                               "Make it snowy",
                               "Make it Minecraft style"],
           "edited_prompts": ["a high-quality, detailed, and professional image of a red car",
                              "a high-quality, detailed, and professional image of a light-blue car in the night",
                              "a high-quality, detailed, and professional image of a light-blue car in the style of Van Gogh",
                              "a high-quality, detailed, and professional image of a light-blue car in the snow",
                              "a high-quality, detailed, and professional image of a light-blue car in Minecraft style"]}

scene_2 = {"path": "/home/danielgilo/3dips/seva/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557",
           "prompt": "a high-quality, detailed, and professional image of a building, with trees, bushes and a street lamp",
           "editing_prompts": ["Remove the street lamp",
                               "Turn the trees to pine trees",
                               "Make it in cartoon style",
                               "Make it in the spring",
                               "add people"],
           "edited_prompts": ["a high-quality, detailed, and professional image of a building, with trees and bushes",
                              "a high-quality, detailed, and professional image of a building, with pine trees, bushes and a street lamp",
                              "a high-quality, detailed, and professional image of a building, with trees, bushes and a street lamp, in cartoon style",
                              "a high-quality, detailed, and professional image of a building, with trees, bushes and a street lamp, in the spring",
                              "a high-quality, detailed, and professional image of a building, with trees, bushes and a street lamp, and people around"]}


scene_3 = {"path": "/home/danielgilo/3dips/seva/assets_demo_cli/garden_flythrough",
           "prompt": "a high-quality, detailed, and professional image of a vase with a plant, on a round table in a garden",
           "editing_prompts": ["Swap the plant with roses",
                               "turn the table to rosewood table",
                               "make it look like it just rained"],
           "edited_prompts": ["a high-quality, detailed, and professional image of a vase with roses, on a round table in a garden",
                              "a high-quality, detailed, and professional image of a vase with a plant, on a round rosewood table in a garden",
                              "a high-quality, detailed, and professional image of a vase with roses, on a round table in a garden, just after it rained"]}


scenes = [scene_1, scene_3]
teachers = ["pix2pix", "canny", "depth"]

for scene in scenes:
    for teacher in teachers:
        if teacher == "pix2pix":
            for editing_prompt, edited_prompt in zip(scene["editing_prompts"], scene["edited_prompts"]):
                cmd = [
                    "python", "-m", "baselines.individual_preds_baseline",
                    "--teacher_name", teacher,
                    "--scene_path", scene["path"],
                    "--prompt", scene["prompt"],
                    "--editing_prompt", editing_prompt,
                    "--edited_prompt", edited_prompt
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd)

                cmd = [
                    "python", "-m", "baselines.sdedit_baseline",
                    "--teacher_name", teacher,
                    "--scene_path", scene["path"],
                    "--prompt", scene["prompt"],
                    "--timestep", "250",
                    "--editing_prompt", editing_prompt,
                    "--edited_prompt", edited_prompt
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd)

                cmd = [
                    "python", "-m", "baselines.sdedit_baseline",
                    "--teacher_name", teacher,
                    "--scene_path", scene["path"],
                    "--prompt", scene["prompt"],
                    "--timestep", "500",
                    "--editing_prompt", editing_prompt,
                    "--edited_prompt", edited_prompt
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd)

                cmd = [
                    "python", "-m", "baselines.sdedit_baseline",
                    "--teacher_name", teacher,
                    "--scene_path", scene["path"],
                    "--prompt", scene["prompt"],
                    "--timestep", "750",
                    "--editing_prompt", editing_prompt,
                    "--edited_prompt", edited_prompt
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd)

                cmd = [
                    "python", "-m", "baselines.text2video-zero_baseline",
                    "--teacher_name", teacher,
                    "--scene_path", scene["path"],
                    "--prompt", scene["prompt"],
                    "--editing_prompt", editing_prompt,
                    "--edited_prompt", edited_prompt
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd)

        else:
            cmd = [
                "python", "-m", "baselines.individual_preds_baseline",
                "--teacher_name", teacher,
                "--scene_path", scene["path"],
                "--prompt", scene["prompt"]
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)

            cmd = [
                "python", "-m", "baselines.sdedit_baseline",
                "--teacher_name", teacher,
                "--scene_path", scene["path"],
                "--prompt", scene["prompt"],
                "--timestep", "250"
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)

            cmd = [
                "python", "-m", "baselines.sdedit_baseline",
                "--teacher_name", teacher,
                "--scene_path", scene["path"],
                "--prompt", scene["prompt"],
                "--timestep", "500"
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)

            cmd = [
                "python", "-m", "baselines.sdedit_baseline",
                "--teacher_name", teacher,
                "--scene_path", scene["path"],
                "--prompt", scene["prompt"],
                "--timestep", "750"
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)

            cmd = [
                "python", "-m", "baselines.text2video-zero_baseline",
                "--teacher_name", teacher,
                "--scene_path", scene["path"],
                "--prompt", scene["prompt"]
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)