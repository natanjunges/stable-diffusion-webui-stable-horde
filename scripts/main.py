# Stable Horde for Web UI, a Stable Horde client for AUTOMATIC1111's Stable Diffusion Web UI
# Copyright (C) 2023  Natan Junges <natanajunges@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from modules import scripts, processing, shared, images, devices, ui, lowvram
import gradio
import requests
import time
import PIL.Image
import base64
import io
import os.path
import numpy
import itertools
import torch
from scripts.settings_manager import SettingsManager

class FakeCheckpointInfo:
    def __init__(self, model_name):
        self.model_name = model_name

class FakeModel:
    sd_model_hash=""

    def __init__(self, name):
        self.sd_checkpoint_info = FakeCheckpointInfo(name)

class StableHordeGenerateError(Exception):
    pass

class Main(SettingsManager, scripts.Script):
    TITLE = "Run on Stable Horde"
    SAMPLERS = {
        "LMS": "k_lms",
        "LMS Karras": "k_lms",
        "Heun": "k_heun",
        "Euler": "k_euler",
        "Euler a": "k_euler_a",
        "DPM2": "k_dpm_2",
        "DPM2 Karras": "k_dpm_2",
        "DPM2 a": "k_dpm_2_a",
        "DPM2 a Karras": "k_dpm_2_a",
        "DPM fast": "k_dpm_fast",
        "DPM adaptive": "k_dpm_adaptive",
        "DPM++ 2S a": "k_dpmpp_2s_a",
        "DPM++ 2S a Karras": "k_dpmpp_2s_a",
        "DPM++ 2M": "k_dpmpp_2m",
        "DPM++ 2M Karras": "k_dpmpp_2m",
        "DPM++ SDE": "k_dpmpp_sde",
        "DPM++ SDE Karras": "k_dpmpp_sde"
    }
    POST_PROCESSINGS = {"CodeFormers (Face restoration)", "GFPGAN (Face restoration)", "RealESRGAN_x4plus (Upscaling)"}

    def title(self):
        return self.TITLE

    def show(self, is_img2img):
        return True

    def load_models(self):
        self.load_settings()

        try:
            models = requests.get("{}/v2/status/models".format(self.api_endpoint), headers={"Client-Agent": self.CLIENT_AGENT})
            models = models.json()
            models.sort(key=lambda m: (-m["count"], m["name"]))
            models = ["{} ({})".format(m["name"], m["count"]) for m in models]
        except requests.ConnectionError:
            models = []

        models.insert(0, "Random")
        self.models = models

    def ui(self, is_img2img):
        with gradio.Box():
            with gradio.Row(elem_id="horde_model_row"):
                self.load_models()
                model = gradio.Dropdown(self.models, value="Random", label="Model")
                model.style(container=False)
                update = gradio.Button(ui.refresh_symbol, elem_id="horde_update_model")

        with gradio.Box():
            with gradio.Row():
                nsfw = gradio.Checkbox(False, label="NSFW")
                shared_laion = gradio.Checkbox(False, label="Share with LAION", interactive=not is_img2img)
                seed_variation = gradio.Slider(minimum=1, maximum=1000, value=1, step=1, label="Seed variation", elem_id="horde_seed_variation")

        with gradio.Box():
            with gradio.Row():
                post_processing_1 = gradio.Dropdown(["None"] + sorted(self.POST_PROCESSINGS), value="None", label="Post processing #1")
                post_processing_1.style(container=False)
                post_processing_2 = gradio.Dropdown(["None"] + sorted(self.POST_PROCESSINGS), value="None", label="Post processing #2", interactive=False)
                post_processing_2.style(container=False)
                post_processing_3 = gradio.Dropdown(["None"] + sorted(self.POST_PROCESSINGS), value="None", label="Post processing #3", interactive=False)
                post_processing_3.style(container=False)

        def update_click():
            self.load_models()
            return gradio.update(choices=self.models, value="Random")

        def post_processing_1_change(value_1):
            return (gradio.update(choices=["None"] + sorted(self.POST_PROCESSINGS - {value_1}), value="None", interactive=value_1 != "None"), gradio.update(choices=["None"] + sorted(self.POST_PROCESSINGS - {value_1}), value="None", interactive=False))

        def post_processing_2_change(value_1, value_2):
            return gradio.update(choices=["None"] + sorted(self.POST_PROCESSINGS - {value_1, value_2}), value="None", interactive=value_2 != "None")

        update.click(fn=update_click, outputs=model)
        post_processing_1.change(fn=post_processing_1_change, inputs=post_processing_1, outputs=[post_processing_2, post_processing_3])
        post_processing_2.change(fn=post_processing_2_change, inputs=[post_processing_1, post_processing_2], outputs=post_processing_3)

        def model_infotext(d):
            if "Model" in d and d["Model"] != "Random":
                try:
                    return next(filter(lambda s: s.startswith("{} (".format(d["Model"])), self.models))
                except StopIteration:
                    pass

            return "Random"

        def post_processing_n_infotext(n):
            def post_processing_infotext(d):
                if "Post processing {}".format(n) in d:
                    try:
                        return next(filter(lambda s: s.startswith("{} (".format(d["Post processing {}".format(n)])), self.POST_PROCESSINGS))
                    except StopIteration:
                        pass

                return "None"

            return post_processing_infotext

        self.infotext_fields = [
            (model, model_infotext),
            (nsfw, "NSFW"),
            (shared_laion, "Share with LAION"),
            (seed_variation, "Seed variation"),
            (post_processing_1, post_processing_n_infotext(1)),
            (post_processing_2, post_processing_n_infotext(2)),
            (post_processing_3, post_processing_n_infotext(3))
        ]
        return [model, nsfw, shared_laion, seed_variation, post_processing_1, post_processing_2, post_processing_3]

    def run(self, p, model, nsfw, shared_laion, seed_variation, post_processing_1, post_processing_2, post_processing_3):
        if model != "Random":
            model = model.split("(")[0].rstrip()

        post_processing = []

        if post_processing_1 != "None":
            post_processing.append(post_processing_1.split("(")[0].rstrip())

            if post_processing_2 != "None":
                post_processing.append(post_processing_2.split("(")[0].rstrip())

                if post_processing_3 != "None":
                    post_processing.append(post_processing_3.split("(")[0].rstrip())

        return self.process_images(p, model, nsfw, shared_laion, int(seed_variation), post_processing)

    def process_images(self, p, model, nsfw, shared_laion, seed_variation, post_processing):
        # Copyright (C) 2023  AUTOMATIC1111
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/d7aec59c4eb02f723b3d55c6f927a42e97acd679/modules/processing.py#L463-L490

        stored_opts = {k: shared.opts.data[k] for k in p.override_settings.keys()}

        try:
            for k, v in p.override_settings.items():
                setattr(shared.opts, k, v)

            p.extra_generation_params = {
                "Model": model,
                "NSFW": nsfw,
                "Share with LAION": self.api_key == "0000000000" or shared_laion if not self.is_img2img else None,
                "Seed variation": seed_variation if p.batch_size > 1 else None,
                "Post processing 1": (post_processing[0] if len(post_processing) >= 1 else None),
                "Post processing 2": (post_processing[1] if len(post_processing) >= 2 else None),
                "Post processing 3": (post_processing[2] if len(post_processing) >= 3 else None)
            }

            res = self.process_images_inner(p, model, nsfw, shared_laion, seed_variation, post_processing)
        finally:
            if p.override_settings_restore_afterwards:
                for k, v in stored_opts.items():
                    setattr(shared.opts, k, v)

        return res

    def process_images_inner(self, p, model, nsfw, shared_laion, seed_variation, post_processing):
        # Copyright (C) 2023  AUTOMATIC1111
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/00dab8f10defbbda579a1bc89c8d4e972c58a20d/modules/processing.py#L501-L717

        fake_model = FakeModel(model)

        if type(p.prompt) == list:
            assert(len(p.prompt) > 0)
        else:
            assert p.prompt is not None

        devices.torch_gc()
        seed = processing.get_fixed_seed(p.seed)
        p.subseed = -1
        p.subseed_strength = 0
        p.seed_resize_from_h = 0
        p.seed_resize_from_w = 0

        if type(p.prompt) == list:
            p.all_prompts = list(itertools.chain.from_iterable((p.batch_size * [shared.prompt_styles.apply_styles_to_prompt(p.prompt[x * p.batch_size], p.styles)] for x in range(p.n_iter))))
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

        if type(p.negative_prompt) == list:
            p.all_negative_prompts = list(itertools.chain.from_iterable((p.batch_size * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt[x * p.batch_size], p.styles)] for x in range(p.n_iter))))
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

        if type(seed) == list:
            p.all_seeds = list(itertools.chain.from_iterable(([seed[x * p.batch_size] + y * seed_variation for y in range(p.batch_size)] for x in range(p.n_iter))))
        else:
            p.all_seeds = [int(seed) + x * seed_variation for x in range(len(p.all_prompts))]

        p.all_subseeds = [-1 for _ in range(len(p.all_prompts))]

        def infotext(iteration=0, position_in_batch=0):
            old_model = shared.sd_model
            shared.sd_model = fake_model
            ret = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, {}, iteration, position_in_batch)
            shared.sd_model = old_model
            return ret

        if p.scripts is not None:
            p.scripts.process(p)

        infotexts = []
        output_images = []

        with torch.no_grad():
            with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
                old_model = shared.sd_model
                shared.sd_model = fake_model
                processed = processing.Processed(p, [], p.seed, "")
                file.write(processed.infotext(p, 0))
                shared.sd_model = old_model

            if shared.state.job_count == -1:
                shared.state.job_count = p.n_iter

            for n in range(p.n_iter):
                p.iteration = n

                if shared.state.skipped:
                    shared.state.skipped = False

                if shared.state.interrupted:
                    break

                prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
                subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

                if len(prompts) == 0:
                    break

                if p.scripts is not None:
                    p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

                if p.n_iter > 1:
                    shared.state.job = f"Batch {n+1} out of {p.n_iter}"

                x_samples_ddim, models = self.process_batch_horde(p, model, nsfw, shared_laion, seed_variation, post_processing, prompts[0], negative_prompts[0], seeds[0])

                if x_samples_ddim is None or len(x_samples_ddim) == 0:
                    break

                x_samples_ddim = [s.cpu() for s in x_samples_ddim]
                x_samples_ddim = torch.stack(x_samples_ddim).float()

                if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                    lowvram.send_everything_to_cpu()

                devices.torch_gc()

                if p.scripts is not None:
                    p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * numpy.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(numpy.uint8)
                    image = PIL.Image.fromarray(x_sample)
                    p.extra_generation_params["Model"] = models[i]

                    if p.scripts is not None:
                        pp = scripts.PostprocessImageArgs(image)
                        p.scripts.postprocess_image(p, pp)
                        image = pp.image

                    if p.color_corrections is not None and i < len(p.color_corrections):
                        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_color_correction:
                            image_without_cc = processing.apply_overlay(image, p.paste_to, i, p.overlay_images)
                            images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")

                        image = processing.apply_color_correction(p.color_corrections[i], image)

                    image = processing.apply_overlay(image, p.paste_to, i, p.overlay_images)

                    if shared.opts.samples_save and not p.do_not_save_samples:
                        images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=infotext(n, i), p=p)

                    text = infotext(n, i)
                    infotexts.append(text)

                    if shared.opts.enable_pnginfo:
                        image.info["parameters"] = text

                    output_images.append(image)

                del x_samples_ddim
                devices.torch_gc()
                p.extra_generation_params["Model"] = model
                shared.state.job_no += 1
                shared.state.sampling_step = 0
                shared.state.current_image_sampling_step = 0

            p.color_corrections = None
            index_of_first_image = 0
            unwanted_grid_because_of_img_count = len(output_images) < 2 and shared.opts.grid_only_if_multiple

            if (shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
                grid = images.image_grid(output_images, p.batch_size)

                if shared.opts.return_grid:
                    text = infotext()
                    infotexts.insert(0, text)

                    if shared.opts.enable_pnginfo:
                        grid.info["parameters"] = text

                    output_images.insert(0, grid)
                    index_of_first_image = 1

                if shared.opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=infotext(), short_filename=not shared.opts.grid_extended_filename, p=p, grid=True)

        devices.torch_gc()
        old_model = shared.sd_model
        shared.sd_model = fake_model
        res = processing.Processed(p, output_images, p.all_seeds[0], infotext(), subseed=-1, index_of_first_image=index_of_first_image, infotexts=infotexts)
        shared.sd_model = old_model

        if p.scripts is not None:
            p.scripts.postprocess(p, res)

        return res

    def process_batch_horde(self, p, model, nsfw, shared_laion, seed_variation, post_processing, prompt, negative_prompt, seed):
        try:
            assert model != "stable_diffusion_inpainting" or self.is_img2img and p.image_mask is not None, "Model stable_diffusion_inpainting can only be used for inpainting"
            assert model not in {"Stable Diffusion 2 Depth", "pix2pix"} or self.is_img2img and p.image_mask is None, "Model {} can only be used for img2img".format(model)
        except AssertionError as e:
            raise StableHordeGenerateError(str(e))

        payload = {
            "prompt": "{} ### {}".format(prompt, negative_prompt) if len(negative_prompt) > 0 else prompt,
            "params": {
                "cfg_scale": p.cfg_scale,
                "seed": str(seed),
                "height": p.height,
                "width": p.width,
                "steps": p.steps,
                "n": p.batch_size
            },
            "r2": False
        }

        self.load_settings()

        if p.batch_size > 1:
            payload["params"]["seed_variation"] = seed_variation

        if len(post_processing) > 0:
            payload["params"]["post_processing"] = post_processing

        if not self.is_img2img or p.image_mask is None:
            if self.is_img2img:
                payload["params"]["denoising_strength"] = p.denoising_strength
            elif p.enable_hr:
                payload["params"]["hires_fix"] = True

            if model != "Stable Diffusion 2 Depth":
                payload["params"]["sampler_name"] = self.SAMPLERS.get(p.sampler_name)

                if payload["params"]["sampler_name"] is None:
                    payload["params"]["sampler_name"] = "k_euler_a"
                    p.extra_generation_params["Sampler"] = "Euler a"
                elif " Karras" in p.sampler_name:
                    payload["params"]["karras"] = True

                if p.tiling:
                    payload["params"]["tiling"] = True
            else:
                p.extra_generation_params["Sampler"] = "Euler a"
        else:
            p.extra_generation_params["Sampler"] = "Euler a"

        if nsfw:
            payload["nsfw"] = True
        elif self.censor_nsfw:
            payload["censor_nsfw"] = True

        if not self.trusted_workers:
            payload["trusted_workers"] = False

        if len(self.workers) > 0:
            payload["workers"] = self.workers

        if model != "Random":
            payload["models"] = [model]

        if self.is_img2img:
            buffer = io.BytesIO()
            p.init_images[0].save(buffer, format="WEBP")
            payload["source_image"] = base64.b64encode(buffer.getvalue()).decode()

            if p.image_mask is not None:
                payload["source_processing"] = "inpainting"
                buffer = io.BytesIO()
                p.image_mask.save(buffer, format="WEBP")
                payload["source_mask"] = base64.b64encode(buffer.getvalue()).decode()

        if not self.is_img2img and self.api_key != "0000000000" and shared_laion:
            payload["shared"] = True

        if shared.state.skipped or shared.state.interrupted:
            return (None, None)

        try:
            session = requests.Session()
            id = session.post("{}/v2/generate/async".format(self.api_endpoint), headers={"apikey": self.api_key, "Client-Agent": self.CLIENT_AGENT}, json=payload)
            assert id.status_code == 202, "Status Code: {} (expected {})".format(id.status_code, 202)
            id = id.json()
            id = id["id"]
            shared.state.sampling_steps = 0
            start = time.time()
            timeout = 1

            while True:
                if shared.state.skipped or shared.state.interrupted:
                    return self.cancel_process_batch_horde(id)

                try:
                    status = session.get("{}/v2/generate/check/{}".format(self.api_endpoint, id), headers={"Client-Agent": self.CLIENT_AGENT}, timeout=timeout)
                    assert status.status_code == 200, "Status Code: {} (expected {})".format(status.status_code, 200)
                    status = status.json()
                    elapsed = int(time.time() - start)
                    shared.state.sampling_steps = elapsed + status["wait_time"]
                    shared.state.sampling_step = elapsed

                    if status["done"]:
                        shared.state.sampling_steps = shared.state.sampling_step
                        images = session.get("{}/v2/generate/status/{}".format(self.api_endpoint, id), headers={"Client-Agent": self.CLIENT_AGENT})
                        images = images.json()
                        images = images["generations"]
                        models = [image["model"] for image in images]
                        images = [PIL.Image.open(io.BytesIO(base64.b64decode(image["img"]))) for image in images]
                        images = [numpy.moveaxis(numpy.array(image).astype(numpy.float32) / 255.0, 2, 0) for image in images]
                        images = [torch.from_numpy(image) for image in images]
                        return (images, models)
                    elif status["faulted"]:
                        raise StableHordeGenerateError("This request caused an internal server error and could not be completed.")
                    elif not status["is_possible"]:
                        raise StableHordeGenerateError("This request will not be able to be completed with the pool of workers currently available.")
                    else:
                        if timeout > 1:
                            timeout //= 2

                        time.sleep(1)
                except requests.Timeout:
                    if timeout >= 60:
                        raise StableHordeGenerateError("Reached maximum number of retries")

                    timeout *= 2
                    time.sleep(1)
                except AssertionError:
                    status = status.json()
                    raise StableHordeGenerateError(status["message"])
        except AssertionError:
            id = id.json()
            raise StableHordeGenerateError(id["message"])

    def cancel_process_batch_horde(self, id):
        images = requests.delete("{}/v2/generate/status/{}".format(self.api_endpoint, id), headers={"Client-Agent": self.CLIENT_AGENT}, timeout=60)
        images = images.json()
        images = images["generations"]
        models = [image["model"] for image in images]
        images = [PIL.Image.open(io.BytesIO(base64.b64decode(image["img"]))) for image in images]
        images = [numpy.moveaxis(numpy.array(image).astype(numpy.float32) / 255.0, 2, 0) for image in images]
        images = [torch.from_numpy(image) for image in images]

        if len(images) > 0:
            return (images, models)
        else:
            return (None, None)
