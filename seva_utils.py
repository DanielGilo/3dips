import torch
import numpy as np
from seva.seva.eval import transform_img_and_K, load_img_and_K, seed_everything, get_value_dict, chunk_input_and_test, pad_indices, get_k_from_dict, assemble
from seva.seva.data_io import get_parser
from seva.seva.eval import (
    compute_relative_inds,
    infer_prior_stats,
)
from seva.seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_lookat,
)

# copied from seva's demo.py
def parse_task(
    task,
    scene,
    num_inputs,
    T,
    version_dict,
):
    options = version_dict["options"]

    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    parser = get_parser(
        parser_type="reconfusion",
        data_dir=scene,
        normalize=False,
    )
    all_imgs_path = parser.image_paths
    c2ws = parser.camtoworlds
    camera_ids = parser.camera_ids
    Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)

    if num_inputs is None:
        assert len(parser.splits_per_num_input_frames.keys()) == 1
        num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
        split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
    elif isinstance(num_inputs, str):
        split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
        num_inputs = int(num_inputs.split("-")[0])  # for example 1_from32
    else:
        split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore

    num_targets = len(split_dict["test_ids"])

    assert task == "img2img"
    # Note in this setting, we should refrain from using all the other camera
    # info except ones from sampled_indices, and most importantly, the order.
    num_anchors = infer_prior_stats(
        T,
        num_inputs,
        num_total_frames=num_targets,
        version_dict=version_dict,
    )

    sampled_indices = np.sort(
        np.array(split_dict["train_ids"] + split_dict["test_ids"])
    )  # we always sort all indices first

    traj_prior = options.get("traj_prior", None)
    if traj_prior == "spiral":
        assert parser.bounds is not None
        anchor_c2ws = generate_spiral_path(
            c2ws[sampled_indices] @ np.diagflat([1, -1, -1, 1]),
            parser.bounds[sampled_indices],
            n_frames=num_anchors + 1,
            n_rots=2,
            zrate=0.5,
            endpoint=False,
        )[1:] @ np.diagflat([1, -1, -1, 1])
    elif traj_prior == "interpolated":
        assert num_inputs > 1
        anchor_c2ws = generate_interpolated_path(
            c2ws[split_dict["train_ids"], :3],
            round((num_anchors + 1) / (num_inputs - 1)),
            endpoint=False,
        )[1 : num_anchors + 1]
    elif traj_prior == "orbit":
        c2ws_th = torch.as_tensor(c2ws)
        lookat = get_lookat(
            c2ws_th[sampled_indices, :3, 3],
            c2ws_th[sampled_indices, :3, 2],
        )
        anchor_c2ws = torch.linalg.inv(
            get_arc_horizontal_w2cs(
                torch.linalg.inv(c2ws_th[split_dict["train_ids"][0]]),
                lookat,
                -F.normalize(
                    c2ws_th[split_dict["train_ids"]][:, :3, 1].mean(0),
                    dim=-1,
                ),
                num_frames=num_anchors + 1,
                endpoint=False,
            )
        ).numpy()[1:, :3]
    else:
        anchor_c2ws = None
    # anchor_Ks is default to be the first from target_Ks

    all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
    c2ws = c2ws[sampled_indices]
    Ks = Ks[sampled_indices]

    # absolute to relative indices
    input_indices = compute_relative_inds(
        sampled_indices,
        np.array(split_dict["train_ids"]),
    )
    anchor_indices = np.arange(
        sampled_indices.shape[0],
        sampled_indices.shape[0] + num_anchors,
    ).tolist()  # the order has no meaning here


    return (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
        (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
    )

@torch.no_grad()
def get_value_dict_of_scene(scene):

    version_dict = {
        "H": 576,
        "W": 576,
        "T": 21,
        "C": 4,
        "f": 8,
        "options": {},
    }
    seed = 42


    num_inputs = 1
    task = "img2img"


    (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        c2ws,
        Ks,
        anchor_c2ws,
        anchor_Ks,
    ) = parse_task(
        task,
        scene,
        num_inputs,
        version_dict["T"],
        version_dict,
    )
    assert num_inputs is not None
    # Create image conditioning.
    image_cond = {
        "img": all_imgs_path,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }
    # Create camera conditioning.
    camera_cond = {
        "c2w": c2ws.clone(),
        "K": Ks.clone(),
        "input_indices": list(range(num_inputs + num_targets)),
    }



    # run_one_scene

    H, W, T, C, F, options = (
            version_dict["H"],
            version_dict["W"],
            version_dict["T"],
            version_dict["C"],
            version_dict["f"],
            version_dict["options"],
        )

    if isinstance(image_cond, str):
        image_cond = {"img": [image_cond]}
    imgs_clip, imgs, img_size = [], [], None
    for i, (img, K) in enumerate(zip(image_cond["img"], camera_cond["K"])):
        img, K = load_img_and_K(img or img_size, None, K=K, device="cpu")  # type: ignore
        img_size = img.shape[-2:]
        if options.get("L_short", -1) == -1:
            img, K = transform_img_and_K(
                img,
                (W, H),
                K=K[None],
                mode=(
                    options.get("transform_input", "crop")
                    if i in image_cond["input_indices"]
                    else options.get("transform_target", "crop")
                ),
                scale=(
                    1.0
                    if i in image_cond["input_indices"]
                    else options.get("transform_scale", 1.0)
                ),
            )
        else:
            downsample = 3
            assert options["L_short"] % F * 2**downsample == 0, (
                "Short side of the image should be divisible by "
                f"F*2**{downsample}={F * 2**downsample}."
            )
            img, K = transform_img_and_K(
                img,
                options["L_short"],
                K=K[None],
                size_stride=F * 2**downsample,
                mode=(
                    options.get("transform_input", "crop")
                    if i in image_cond["input_indices"]
                    else options.get("transform_target", "crop")
                ),
                scale=(
                    1.0
                    if i in image_cond["input_indices"]
                    else options.get("transform_scale", 1.0)
                ),
            )
            version_dict["W"] = W = img.shape[-1]
            version_dict["H"] = H = img.shape[-2]
        K = K[0]
        K[0] /= W
        K[1] /= H
        camera_cond["K"][i] = K
        img_clip = img
        imgs_clip.append(img_clip)
        imgs.append(img)
    imgs_clip = torch.cat(imgs_clip, dim=0)
    imgs = torch.cat(imgs, dim=0)

    options["num_frames"] = T
    torch.cuda.empty_cache()

    seed_everything(seed)

    # Get Data
    input_indices = image_cond["input_indices"]
    input_imgs = imgs[input_indices]
    input_imgs_clip = imgs_clip[input_indices]
    input_c2ws = camera_cond["c2w"][input_indices]
    input_Ks = camera_cond["K"][input_indices]

    test_indices = [i for i in range(len(imgs)) if i not in input_indices]
    test_imgs = imgs[test_indices]
    test_imgs_clip = imgs_clip[test_indices]
    test_c2ws = camera_cond["c2w"][test_indices]
    test_Ks = camera_cond["K"][test_indices]


    chunk_strategy = options.get("chunk_strategy", "gt")
    (
        _,
        input_inds_per_chunk,
        input_sels_per_chunk,
        test_inds_per_chunk,
        test_sels_per_chunk,
    ) = chunk_input_and_test(
        T,
        input_c2ws,
        test_c2ws,
        input_indices,
        test_indices,
        options=options,
        task=task,
        chunk_strategy=chunk_strategy,
        gt_input_inds=list(range(input_c2ws.shape[0])),
    )
    print(
        f"One pass - chunking with `{chunk_strategy}` strategy: total "
        f"{len(input_inds_per_chunk)} forward(s) ..."
    )

    all_samples = {}
    all_test_inds = []

    chunk_input_inds = input_inds_per_chunk[0]
    chunk_input_sels = input_sels_per_chunk[0]
    chunk_test_inds = test_inds_per_chunk[0]
    chunk_test_sels = test_sels_per_chunk[0]


    (curr_input_sels, curr_test_sels,  curr_input_maps, curr_test_maps) = pad_indices(
        chunk_input_sels,
        chunk_test_sels,
        T=T,
        padding_mode=options.get("t_padding_mode", "last"))
    curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
        assemble(
            input=x[chunk_input_inds],
            test=y[chunk_test_inds],
            input_maps=curr_input_maps,
            test_maps=curr_test_maps,
        )
        for x, y in zip(
            [
                torch.cat(
                    [
                        input_imgs,
                        get_k_from_dict(all_samples, "samples-rgb").to(
                            input_imgs.device
                        ),
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        input_imgs_clip,
                        get_k_from_dict(all_samples, "samples-rgb").to(
                            input_imgs.device
                        ),
                    ],
                    dim=0,
                ),
                torch.cat([input_c2ws, test_c2ws[all_test_inds]], dim=0),
                torch.cat([input_Ks, test_Ks[all_test_inds]], dim=0),
            ],  # procedually append generated prior views to the input views
            [test_imgs, test_imgs_clip, test_c2ws, test_Ks],
        )
    ]
    value_dict = get_value_dict(
        curr_imgs.to("cuda"),
        curr_imgs_clip.to("cuda"),
        curr_input_sels
        + [
            sel
            for (ind, sel) in zip(
                np.array(chunk_test_inds)[curr_test_maps[curr_test_maps != -1]],
                curr_test_sels,
            )
            if test_indices[ind] in image_cond["input_indices"]
        ],
        curr_c2ws,
        curr_Ks,
        curr_input_sels
        + [
            sel
            for (ind, sel) in zip(
                np.array(chunk_test_inds)[curr_test_maps[curr_test_maps != -1]],
                curr_test_sels,
            )
            if test_indices[ind] in camera_cond["input_indices"]
        ],
        all_c2ws=camera_cond["c2w"],
        camera_scale=options.get("camera_scale", 2.0),
    )

    return value_dict
