import gradio as gr


def ui():
    from modules.predict import WaifuFilter
    scorer = WaifuFilter()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    label='image',
                    type='pil',
                    height=512,
                    sources=['upload', 'clipboard'],
                )
            with gr.Column():
                with gr.Row():
                    score = gr.Number(
                        label='filter level',
                    )

        image.change(
            fn=lambda image: scorer.predict([image]*2)[0] if image is not None else None,
            inputs=image,
            outputs=score,
        )

    return demo


def launch():
    demo = ui()
    demo.launch(share=False)
