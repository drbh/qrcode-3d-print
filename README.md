# 3D printable QR code generator

This is a simple QR code generator that generates a 3D printable QR code. The QR code is generated using the `qrcode` library and the 3D model is generated using the `build123d` library (which uses opencascade under the hood).


## Getting started

this project assumes the use of VSCode to enable interactive CAD development. Ensure that you have the OCP CAD Viewer extension installed in VSCode https://marketplace.visualstudio.com/items?itemName=bernhard-42.ocp-cad-viewer.

get the dependencies

```bash
uv sync
```

now start the OCP viewer in VSCode (make to use the virtual environment python interpreter)

edit the `main.py` file 

```python
# this format enables phones to connect wifi by scanning the qr code
wifi_name = "hello"
wifi_password = "world"
text = f"WIFI:S:{wifi_name};T:WPA2;P:{wifi_password};;"
```

```bash
uv run main.py
```

this should output a `qr_code_3d.gltf` file, which can be used in any 3D viewer/downstream application, and the QR code should be visible in the OCP viewer in VSCode.

<img width="1418" alt="ocpviewer" src="https://github.com/user-attachments/assets/06157e6d-3d44-48f1-9bff-439c6e6fb90c" />


## Output

<img width="800" alt="qrcode" src="https://github.com/user-attachments/assets/ccbbb919-05d3-4a21-b823-6dc6251d0370" />
