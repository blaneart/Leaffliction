from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import PySimpleGUI as sg
import cv2
import sys

path = sys.argv[1]
sg.theme('light green')


image = cv2.resize(cv2.imread(path), (256,256))
imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto

layout = 	[
		[sg.Image(data=imgbytes), sg.Image(data=imgbytes), sg.Image(data=imgbytes, key='image')],
		[sg.Text('===   DL classification   ===')],
		[sg.Text('Update Image'), sg.In(r'image',size=(40,1), key='yolo'), sg.FileBrowse()],
		[sg.OK('Update'), sg.Button('Quit')]
			]

window = sg.Window('Prediction',
                   default_element_size=(14,1),
                   text_justification='center',
                   auto_size_text=True).Layout(layout)

while True:
	event, values = window.read()
	img = cv2.imencode('.png', cv2.resize(cv2.imread( values['yolo']), (256,256)))[1].tobytes()
    # See if user wants to quit or window was closed
	if event == sg.WINDOW_CLOSED or event == 'Quit':
		break
    # Output a message to the window
	window['image'].update(data=img)


window.close()
