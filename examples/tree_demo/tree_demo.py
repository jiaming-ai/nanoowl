import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
import weakref
import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from typing import List
from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    IMAGE_QUALITY = args.image_quality

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )

    prompt_data = None

    def get_colors(count: int):
        cmap = plt.cm.get_cmap("rainbow", count)
        colors = []
        for i in range(count):
            color = cmap(i)
            color = [int(255 * value) for value in color]
            colors.append(tuple(color))
        return colors

    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)

    def process_image(image_data):
        if image_data is None:
            logging.info("No image data to process")
            return None

        logging.info("Processing image of length: {}".format(len(image_data)))
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("Failed to decode image")
            return None

        image_pil = cv2_to_pil(image)

        if prompt_data is not None:
            prompt_data_local = prompt_data
            detections = predictor.predict(
                image_pil,
                tree=prompt_data_local['tree'],
                clip_text_encodings=prompt_data_local['clip_encodings'],
                owl_text_encodings=prompt_data_local['owl_encodings']
            )
            image = draw_tree_output(image, detections, prompt_data_local['tree'])

        image_jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1].tobytes()
        logging.info("Processed image of length: {}".format(len(image_jpeg)))
        return image_jpeg

    async def handle_index_get(request: web.Request):
        logging.info("handle_index_get")
        return web.FileResponse("./index.html")

    async def websocket_handler(request):
        global prompt_data

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        logging.info("Websocket connected.")
        request.app['websockets'].add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    if "prompt" in msg.data:
                        header, prompt = msg.data.split(":", 1)
                        logging.info("Received prompt: " + prompt)
                        try:
                            tree = Tree.from_prompt(prompt)
                            clip_encodings = predictor.encode_clip_text(tree)
                            owl_encodings = predictor.encode_owl_text(tree)
                            prompt_data = {
                                "tree": tree,
                                "clip_encodings": clip_encodings,
                                "owl_encodings": owl_encodings
                            }
                            logging.info("Set prompt: " + prompt)
                        except Exception as e:
                            logging.error(e)
                elif msg.type == web.WSMsgType.BINARY:
                    logging.info("Received binary message of length: " + str(len(msg.data)))
                    processed_image = await asyncio.get_running_loop().run_in_executor(None, process_image, msg.data)
                    if processed_image:
                        await ws.send_bytes(processed_image)

        finally:
            request.app['websockets'].discard(ws)
            await ws.close()

        return ws

    async def on_shutdown(app: web.Application):
        for ws in set(app['websockets']):
            await ws.close(code=WSCloseCode.GOING_AWAY, message='Server shutdown')

    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host=args.host, port=args.port)
