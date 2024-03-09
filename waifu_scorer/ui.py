import gradio as gr
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='Eugeoter/waifu-scorer-v2/waifu-scorer-v2-1.pth',
        help='Path or url to the model file',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='mlp',
        help='Type of the model',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use',
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Share the demo',
    )
    return parser.parse_args()


def ui(args):
    from waifu_scorer.predict import WaifuScorer, load_model
    scorer = WaifuScorer(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
    )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    label='Image',
                    type='pil',
                    height=512,
                    sources=['upload', 'clipboard'],
                )
            with gr.Column():
                with gr.Row():
                    model_path = gr.Textbox(
                        label='Model Path',
                        value=args.model_path,
                        placeholder='Path or URL to the model file',
                    )
                with gr.Row():
                    score = gr.Number(
                        label='Score',
                    )

        def change_model(model_path):
            nonlocal scorer
            scorer.mlp = load_model(model_path, model_type=args.model_type, device=args.device)
            print(f"Model changed to `{model_path}`")
            return gr.update()

        model_path.submit(
            fn=change_model,
            inputs=model_path,
            outputs=model_path,
        )

        image.change(
            fn=lambda image: scorer.predict([image]*2)[0] if image is not None else None,
            inputs=image,
            outputs=score,
        )

    return demo


def launch(args):
    demo = ui(args)
    demo.launch(share=args.share)
