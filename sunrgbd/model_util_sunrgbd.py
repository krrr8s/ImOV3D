# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_class = 365 #change this 
        self.num_heading_bin = 12
        self.num_size_cluster = 365 #change this 
        #training sets
        self.type2class={"human": 0,
                            "sneakers": 1,
                            "chair": 2,
                            "hat": 3,
                            "lamp": 4,
                            "bottle": 5,
                            "cabinet/shelf": 6,
                            "cup": 7,
                            "car": 8,
                            "glasses": 9,
                            "picture/frame": 10,
                            "desk": 11,
                            "handbag": 12,
                            "street lights": 13,
                            "book": 14,
                            "plate": 15,
                            "helmet": 16,
                            "leather shoes": 17,
                            "pillow": 18,
                            "glove": 19,
                            "potted plant": 20,
                            "bracelet": 21,
                            "flower": 22,
                            "monitor": 23,
                            "storage box": 24,
                            "plants pot/vase": 25,
                            "bench": 26,
                            "wine glass": 27,
                            "boots": 28,
                            "dining table": 29,
                            "umbrella": 30,
                            "boat": 31,
                            "flag": 32,
                            "speaker": 33,
                            "trash bin/can": 34,
                            "stool": 35,
                            "backpack": 36,
                            "sofa": 37,
                            "belt": 38,
                            "carpet": 39,
                            "basket": 40,
                            "towel/napkin": 41,
                            "slippers": 42,
                            "bowl": 43,
                            "barrel/bucket": 44,
                            "coffee table": 45,
                            "suv": 46,
                            "toy": 47,
                            "tie": 48,
                            "bed": 49,
                            "traffic light": 50,
                            "pen/pencil": 51,
                            "microphone": 52,
                            "sandals": 53,
                            "canned": 54,
                            "necklace": 55,
                            "mirror": 56,
                            "faucet": 57,
                            "bicycle": 58,
                            "bread": 59,
                            "high heels": 60,
                            "ring": 61,
                            "van": 62,
                            "watch": 63,
                            "combine with bowl": 64,
                            "sink": 65,
                            "horse": 66,
                            "fish": 67,
                            "apple": 68,
                            "traffic sign": 69,
                            "camera": 70,
                            "candle": 71,
                            "stuffed animal": 72,
                            "cake": 73,
                            "motorbike/motorcycle": 74,
                            "wild bird": 75,
                            "laptop": 76,
                            "knife": 77,
                            "cellphone": 78,
                            "paddle": 79,
                            "truck": 80,
                            "cow": 81,
                            "power outlet": 82,
                            "clock": 83,
                            "drum": 84,
                            "fork": 85,
                            "bus": 86,
                            "hanger": 87,
                            "nightstand": 88,
                            "pot/pan": 89,
                            "sheep": 90,
                            "guitar": 91,
                            "traffic cone": 92,
                            "tea pot": 93,
                            "keyboard": 94,
                            "tripod": 95,
                            "hockey stick": 96,
                            "fan": 97,
                            "dog": 98,
                            "spoon": 99,
                            "blackboard/whiteboard": 100,
                            "balloon": 101,
                            "air conditioner": 102,
                            "cymbal": 103,
                            "mouse": 104,
                            "telephone": 105,
                            "pickup truck": 106,
                            "orange": 107,
                            "banana": 108,
                            "airplane": 109,
                            "luggage": 110,
                            "skis": 111,
                            "soccer": 112,
                            "trolley": 113,
                            "oven": 114,
                            "remote": 115,
                            "combine with glove": 116,
                            "paper towel": 117,
                            "refrigerator": 118,
                            "train": 119,
                            "tomato": 120,
                            "machinery vehicle": 121,
                            "tent": 122,
                            "shampoo/shower gel": 123,
                            "head phone": 124,
                            "lantern": 125,
                            "donut": 126,
                            "cleaning products": 127,
                            "sailboat": 128,
                            "tangerine": 129,
                            "pizza": 130,
                            "kite": 131,
                            "computer box": 132,
                            "elephant": 133,
                            "toiletries": 134,
                            "gas stove": 135,
                            "broccoli": 136,
                            "toilet": 137,
                            "stroller": 138,
                            "shovel": 139,
                            "baseball bat": 140,
                            "microwave": 141,
                            "skateboard": 142,
                            "surfboard": 143,
                            "surveillance camera": 144,
                            "gun": 145,
                            "Life saver": 146,
                            "cat": 147,
                            "lemon": 148,
                            "liquid soap": 149,
                            "zebra": 150,
                            "duck": 151,
                            "sports car": 152,
                            "giraffe": 153,
                            "pumpkin": 154,
                            "Accordion/keyboard/piano": 155,
                            "radiator": 156,
                            "converter": 157,
                            "tissue": 158,
                            "carrot": 159,
                            "washing machine": 160,
                            "vent": 161,
                            "cookies": 162,
                            "cutting/chopping board": 163,
                            "tennis racket": 164,
                            "candy": 165,
                            "skating and skiing shoes": 166,
                            "scissors": 167,
                            "folder": 168,
                            "baseball": 169,
                            "strawberry": 170,
                            "bow tie": 171,
                            "pigeon": 172,
                            "pepper": 173,
                            "coffee machine": 174,
                            "bathtub": 175,
                            "snowboard": 176,
                            "suitcase": 177,
                            "grapes": 178,
                            "ladder": 179,
                            "pear": 180,
                            "american football": 181,
                            "basketball": 182,
                            "potato": 183,
                            "paint brush": 184,
                            "printer": 185,
                            "billiards": 186,
                            "fire hydrant": 187,
                            "goose": 188,
                            "projector": 189,
                            "sausage": 190,
                            "fire extinguisher": 191,
                            "extension cord": 192,
                            "facial mask": 193,
                            "tennis ball": 194,
                            "chopsticks": 195,
                            "Electronic stove and gas st": 196,
                            "pie": 197,
                            "frisbee": 198,
                            "kettle": 199,
                            "hamburger": 200,
                            "golf club": 201,
                            "cucumber": 202,
                            "clutch": 203,
                            "blender": 204,
                            "tong": 205,
                            "slide": 206,
                            "hot dog": 207,
                            "toothbrush": 208,
                            "facial cleanser": 209,
                            "mango": 210,
                            "deer": 211,
                            "egg": 212,
                            "violin": 213,
                            "marker": 214,
                            "ship": 215,
                            "chicken": 216,
                            "onion": 217,
                            "ice cream": 218,
                            "tape": 219,
                            "wheelchair": 220,
                            "plum": 221,
                            "bar soap": 222,
                            "scale": 223,
                            "watermelon": 224,
                            "cabbage": 225,
                            "router/modem": 226,
                            "golf ball": 227,
                            "pine apple": 228,
                            "crane": 229,
                            "fire truck": 230,
                            "peach": 231,
                            "cello": 232,
                            "notepaper": 233,
                            "tricycle": 234,
                            "toaster": 235,
                            "helicopter": 236,
                            "green beans": 237,
                            "brush": 238,
                            "carriage": 239,
                            "cigar": 240,
                            "earphone": 241,
                            "penguin": 242,
                            "hurdle": 243,
                            "swing": 244,
                            "radio": 245,
                            "CD": 246,
                            "parking meter": 247,
                            "swan": 248,
                            "garlic": 249,
                            "french fries": 250,
                            "horn": 251,
                            "avocado": 252,
                            "saxophone": 253,
                            "trumpet": 254,
                            "sandwich": 255,
                            "cue": 256,
                            "kiwi fruit": 257,
                            "bear": 258,
                            "fishing rod": 259,
                            "cherry": 260,
                            "tablet": 261,
                            "green vegetables": 262,
                            "nuts": 263,
                            "corn": 264,
                            "key": 265,
                            "screwdriver": 266,
                            "globe": 267,
                            "broom": 268,
                            "pliers": 269,
                            "hammer": 270,
                            "volleyball": 271,
                            "eggplant": 272,
                            "trophy": 273,
                            "board eraser": 274,
                            "dates": 275,
                            "rice": 276,
                            "tape measure/ruler": 277,
                            "dumbbell": 278,
                            "hamimelon": 279,
                            "stapler": 280,
                            "camel": 281,
                            "lettuce": 282,
                            "goldfish": 283,
                            "meat balls": 284,
                            "medal": 285,
                            "toothpaste": 286,
                            "antelope": 287,
                            "shrimp": 288,
                            "rickshaw": 289,
                            "trombone": 290,
                            "pomegranate": 291,
                            "coconut": 292,
                            "jellyfish": 293,
                            "mushroom": 294,
                            "calculator": 295,
                            "treadmill": 296,
                            "butterfly": 297,
                            "egg tart": 298,
                            "cheese": 299,
                            "pomelo": 300,
                            "pig": 301,
                            "race car": 302,
                            "rice cooker": 303,
                            "tuba": 304,
                            "crosswalk sign": 305,
                            "papaya": 306,
                            "hair dryer": 307,
                            "green onion": 308,
                            "chips": 309,
                            "dolphin": 310,
                            "sushi": 311,
                            "urinal": 312,
                            "donkey": 313,
                            "electric drill": 314,
                            "spring rolls": 315,
                            "tortoise/turtle": 316,
                            "parrot": 317,
                            "flute": 318,
                            "measuring cup": 319,
                            "shark": 320,
                            "steak": 321,
                            "poker card": 322,
                            "binoculars": 323,
                            "llama": 324,
                            "radish": 325,
                            "noodles": 326,
                            "mop": 327,
                            "yak": 328,
                            "crab": 329,
                            "microscope": 330,
                            "barbell": 331,
                            "Bread/bun": 332,
                            "baozi": 333,
                            "lion": 334,
                            "red cabbage": 335,
                            "polar bear": 336,
                            "lighter": 337,
                            "mangosteen": 338,
                            "seal": 339,
                            "comb": 340,
                            "eraser": 341,
                            "pitaya": 342,
                            "scallop": 343,
                            "pencil case": 344,
                            "saw": 345,
                            "table tennis  paddle": 346,
                            "okra": 347,
                            "starfish": 348,
                            "monkey": 349,
                            "eagle": 350,
                            "durian": 351,
                            "rabbit": 352,
                            "game board": 353,
                            "french horn": 354,
                            "ambulance": 355,
                            "asparagus": 356,
                            "hoverboard": 357,
                            "pasta": 358,
                            "target": 359,
                            "hotair balloon": 360,
                            "chainsaw": 361,
                            "lobster": 362,
                            "iron": 363,
                            "flashlight": 364,}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        #change this 
        self.type_mean_size = {"Accordion/keyboard/piano": np.array([0.784012,0.380423,0.212827]),
"Bread/bun": np.array([0.201084,0.093798,0.090671]),
"CD": np.array([0.171086,0.118116,0.058805]),
"Electronic stove and gas st": np.array([0.81754,0.520159,0.24115]),
"air conditioner": np.array([0.624328,0.247327,0.387311]),
"airplane": np.array([0.999455,0.591397,0.489153]),
"antelope": np.array([0.330495,0.171326,0.06773]),
"apple": np.array([0.140607,0.068009,0.059628]),
"backpack": np.array([0.571448,0.375673,0.350425]),
"balloon": np.array([0.307896,0.16837,0.30322]),
"banana": np.array([0.263661,0.09647,0.097885]),
"bar soap": np.array([0.124312,0.068461,0.033674]),
"barbell": np.array([0.153054,0.081414,0.100286]),
"barrel/bucket": np.array([0.425749,0.223087,0.310847]),
"baseball bat": np.array([0.36378,0.115955,0.406609]),
"basket": np.array([0.4366,0.265924,0.203811]),
"bathtub": np.array([1.092982,0.621349,0.486559]),
"bed": np.array([1.935493,1.089558,0.731659]),
"belt": np.array([0.324371,0.156587,0.066703]),
"bench": np.array([1.304001,0.559556,0.517296]),
"bicycle": np.array([1.715897,0.572218,0.988121]),
"binoculars": np.array([0.467475,0.162218,0.124677]),
"blackboard/whiteboard": np.array([1.425404,0.196649,0.866886]),
"blender": np.array([0.403089,0.224656,0.292586]),
"board eraser": np.array([0.156967,0.067825,0.034136]),
"boat": np.array([0.468295,0.0452,0.259935]),
"book": np.array([0.191912,0.063688,0.203041]),
"boots": np.array([0.380533,0.146658,0.227806]),
"bottle": np.array([0.284468,0.074083,0.187204]),
"bow tie": np.array([0.219319,0.112291,0.161746]),
"bowl": np.array([0.314163,0.164012,0.093049]),
"bread": np.array([0.300732,0.100356,0.305159]),
"broccoli": np.array([0.140996,0.103226,0.030336]),
"broom": np.array([0.446064,0.138327,0.692184]),
"brush": np.array([0.225479,0.08024,0.120034]),
"butterfly": np.array([0.341243,0.049422,0.258576]),
"cabinet/shelf": np.array([0.577567,0.138825,0.65536]),
"cake": np.array([0.400603,0.297745,0.103506]),
"calculator": np.array([0.286506,0.130596,0.065468]),
"camel": np.array([0.105611,0.010813,0.101571]),
"camera": np.array([0.229664,0.138931,0.099493]),
"candle": np.array([0.169637,0.074132,0.112091]),
"candy": np.array([0.082981,0.039518,0.044765]),
"canned": np.array([0.177409,0.081462,0.142956]),
"car": np.array([0.27998,0.061179,0.10982]),
"carpet": np.array([0.95583,0.547098,0.102245]),
"cat": np.array([0.364939,0.21983,0.198052]),
"cello": np.array([0.942617,0.250269,0.855423]),
"cellphone": np.array([0.152849,0.095411,0.034553]),
"chair": np.array([0.73801,0.400672,0.473042]),
"cherry": np.array([0.762339,0.096504,0.194752]),
"chopsticks": np.array([0.236266,0.108707,0.106308]),
"clock": np.array([0.245232,0.081748,0.138411]),
"coconut": np.array([0.197278,0.056634,0.234106]),
"coffee machine": np.array([0.633601,0.327642,0.375311]),
"coffee table": np.array([1.052394,0.634902,0.484992]),
"comb": np.array([0.289998,0.154634,0.063494]),
"combine with bowl": np.array([0.778964,0.542459,0.243732]),
"computer box": np.array([0.464168,0.240341,0.255933]),
"cookies": np.array([0.068763,0.054635,0.017732]),
"crosswalk sign": np.array([0.77399,0.098137,0.404358]),
"cup": np.array([0.242382,0.090923,0.113842]),
"cutting/chopping board": np.array([0.443354,0.293741,0.083466]),
"cymbal": np.array([0.351914,0.108405,0.154564]),
"desk": np.array([1.303899,0.684984,0.521689]),
"dining table": np.array([1.288557,0.719448,0.33791]),
"dog": np.array([0.648805,0.348722,0.258442]),
"dolphin": np.array([0.497086,0.097725,0.354464]),
"donut": np.array([0.090553,0.068319,0.033998]),
"drum": np.array([0.812624,0.525189,0.53056]),
"duck": np.array([0.393961,0.11825,0.190006]),
"dumbbell": np.array([0.08509,0.048957,0.092945]),
"eagle": np.array([0.208006,0.038647,0.150416]),
"earphone": np.array([0.285185,0.144917,0.073801]),
"egg": np.array([0.07117,0.036652,0.069111]),
"egg tart": np.array([0.132487,0.110773,0.06011]),
"electric drill": np.array([0.352862,0.152059,0.107386]),
"elephant": np.array([0.733148,0.109983,0.454991]),
"eraser": np.array([0.118613,0.04783,0.025261]),
"extension cord": np.array([0.357771,0.1275,0.08895]),
"facial cleanser": np.array([0.229352,0.131167,0.240917]),
"fan": np.array([0.54309,0.26475,0.398798]),
"faucet": np.array([0.299045,0.107701,0.138009]),
"fire extinguisher": np.array([0.187706,0.08752,0.23135]),
"fire hydrant": np.array([0.244225,0.195457,0.499402]),
"fish": np.array([0.156126,0.049646,0.115099]),
"fishing rod": np.array([0.255693,0.011455,0.238448]),
"flag": np.array([0.540279,0.138516,0.661368]),
"flashlight": np.array([0.22372,0.081327,0.151308]),
"flower": np.array([0.380655,0.096914,0.14925]),
"flute": np.array([0.145238,0.100691,0.520483]),
"folder": np.array([0.485276,0.234083,0.308024]),
"fork": np.array([0.143389,0.044949,0.046477]),
"french horn": np.array([0.285281,0.115503,0.167302]),
"frisbee": np.array([0.356423,0.332779,0.036555]),
"game board": np.array([0.631835,0.381173,0.110856]),
"garlic": np.array([0.084297,0.033265,0.036215]),
"giraffe": np.array([0.643127,0.5419,1.16418]),
"glasses": np.array([0.118567,0.071663,0.030831]),
"globe": np.array([0.386978,0.210735,0.226308]),
"glove": np.array([0.258936,0.109056,0.129515]),
"goldfish": np.array([0.13572,0.097665,0.063876]),
"golf ball": np.array([0.129159,0.064823,0.075144]),
"golf club": np.array([0.348511,0.15579,0.800518]),
"grapes": np.array([0.056631,0.019976,0.042645]),
"guitar": np.array([0.452507,0.176736,0.529689]),
"gun": np.array([0.620791,0.214062,0.134638]),
"hair dryer": np.array([0.3315,0.159731,0.190232]),
"hammer": np.array([0.266355,0.076737,0.124534]),
"handbag": np.array([0.434591,0.258413,0.218004]),
"hanger": np.array([0.383652,0.090114,0.160526]),
"hat": np.array([0.260976,0.158444,0.138235]),
"head phone": np.array([0.294022,0.160395,0.074774]),
"helmet": np.array([0.2567,0.260338,0.158443]),
"high heels": np.array([0.25157,0.128457,0.08065]),
"hockey stick": np.array([0.521262,0.1021,0.417902]),
"horse": np.array([0.310346,0.056118,0.430625]),
"hotair balloon": np.array([0.624803,0.372312,0.297089]),
"hoverboard": np.array([0.93366,0.378587,0.231614]),
"human": np.array([1.072115,0.458721,0.897902]),
"iron": np.array([0.263302,0.269905,0.332455]),
"kettle": np.array([0.430374,0.21312,0.220918]),
"key": np.array([0.079819,0.044081,0.021985]),
"keyboard": np.array([0.525808,0.251309,0.075957]),
"kite": np.array([0.353776,0.087573,0.365526]),
"kiwi fruit": np.array([0.124167,0.057878,0.046511]),
"knife": np.array([0.175462,0.052927,0.09041]),
"ladder": np.array([0.842767,0.405668,1.06686]),
"lamp": np.array([0.50073,0.263144,0.485588]),
"lantern": np.array([0.447922,0.210006,0.205519]),
"laptop": np.array([0.641438,0.415675,0.252372]),
"leather shoes": np.array([0.37251,0.166494,0.125548]),
"lemon": np.array([0.149329,0.071654,0.046454]),
"lighter": np.array([0.167144,0.06581,0.074049]),
"lion": np.array([0.37878,0.064785,0.247284]),
"liquid soap": np.array([0.228451,0.076764,0.176507]),
"luggage": np.array([0.685903,0.313701,0.440662]),
"mango": np.array([0.177358,0.09517,0.083239]),
"mangosteen": np.array([0.134946,0.07082,0.054424]),
"marker": np.array([0.155965,0.047802,0.027027]),
"measuring cup": np.array([0.277825,0.102898,0.154249]),
"microphone": np.array([0.358682,0.097669,0.130721]),
"microscope": np.array([0.614637,0.192838,0.319728]),
"microwave": np.array([0.64166,0.360654,0.317565]),
"mirror": np.array([1.250806,0.364554,0.763434]),
"monkey": np.array([0.126011,0.100585,0.146713]),
"mop": np.array([0.392312,0.199002,0.696301]),
"mouse": np.array([0.173636,0.090776,0.050973]),
"necklace": np.array([0.362783,0.10515,0.346578]),
"nightstand": np.array([0.681307,0.519186,0.606855]),
"notepaper": np.array([0.277931,0.150856,0.033556]),
"oven": np.array([0.739336,0.286873,0.652299]),
"paint brush": np.array([0.144684,0.048444,0.224734]),
"paper towel": np.array([0.270081,0.131047,0.203486]),
"parking meter": np.array([0.460519,0.322075,1.038771]),
"parrot": np.array([0.255805,0.513996,0.238351]),
"peach": np.array([0.175705,0.066814,0.042321]),
"pear": np.array([0.27038,0.072249,0.122213]),
"pen/pencil": np.array([0.150064,0.036493,0.022955]),
"pencil case": np.array([0.438672,0.153743,0.067324]),
"penguin": np.array([0.081295,0.033397,0.109749]),
"pepper": np.array([0.190938,0.072264,0.130245]),
"pickup truck": np.array([0.566897,0.037364,0.329211]),
"picture/frame": np.array([0.397442,0.120731,0.26673]),
"pie": np.array([0.474238,0.241584,0.087361]),
"pig": np.array([0.152521,0.112665,0.057148]),
"pillow": np.array([0.609154,0.351014,0.284556]),
"pine apple": np.array([0.30941,0.098225,0.300508]),
"pizza": np.array([0.845684,0.816072,0.470169]),
"plants pot/vase": np.array([0.300344,0.140022,0.182808]),
"plate": np.array([0.30482,0.21491,0.058524]),
"pliers": np.array([0.204393,0.088657,0.139559]),
"poker card": np.array([0.147054,0.08962,0.077603]),
"pomegranate": np.array([0.102959,0.042435,0.06512]),
"pomelo": np.array([0.15599,0.063958,0.062714]),
"pot/pan": np.array([0.517957,0.236956,0.151926]),
"potato": np.array([0.099359,0.058797,0.033011]),
"potted plant": np.array([0.577997,0.30305,0.408925]),
"power outlet": np.array([0.122352,0.077465,0.117632]),
"printer": np.array([0.698096,0.493522,0.320169]),
"projector": np.array([0.691207,0.378933,0.204452]),
"pumpkin": np.array([0.213771,0.084054,0.125954]),
"radiator": np.array([0.717996,0.206041,0.48765]),
"radio": np.array([0.41285,0.236338,0.141844]),
"refrigerator": np.array([0.822395,0.315938,1.067765]),
"remote": np.array([0.235807,0.108147,0.050258]),
"rice cooker": np.array([0.553762,0.310778,0.308411]),
"ring": np.array([0.074425,0.054903,0.008458]),
"router/modem": np.array([0.531633,0.210352,0.186614]),
"sandals": np.array([0.235026,0.184067,0.04928]),
"sandwich": np.array([0.233056,0.117181,0.078299]),
"sausage": np.array([0.357731,0.058998,0.109404]),
"saw": np.array([0.40842,0.120849,0.157029]),
"scale": np.array([0.381378,0.252483,0.090431]),
"scallop": np.array([0.17417,0.058525,0.140524]),
"scissors": np.array([0.166908,0.071367,0.033774]),
"screwdriver": np.array([0.185947,0.057157,0.046684]),
"shampoo/shower gel": np.array([0.15347,0.05285,0.158181]),
"shark": np.array([0.21064,0.084278,0.114138]),
"sheep": np.array([1.566149,0.725485,0.682151]),
"sink": np.array([0.611374,0.413827,0.160779]),
"skateboard": np.array([0.747972,0.239978,0.262216]),
"skating and skiing shoes": np.array([0.346449,0.163901,0.132302]),
"slippers": np.array([0.271409,0.143625,0.102095]),
"snowboard": np.array([1.223055,0.104645,1.150827]),
"sofa": np.array([1.450835,0.831701,0.744989]),
"speaker": np.array([0.364684,0.136416,0.218819]),
"spoon": np.array([0.174538,0.056884,0.098764]),
"stapler": np.array([0.301295,0.109687,0.085788]),
"starfish": np.array([0.11662,0.093103,0.074478]),
"stool": np.array([0.794648,0.431967,0.51812]),
"storage box": np.array([0.436432,0.249392,0.228645]),
"strawberry": np.array([0.164348,0.070041,0.07927]),
"stroller": np.array([0.835815,0.398096,0.750797]),
"stuffed animal": np.array([0.42722,0.190893,0.240832]),
"suitcase": np.array([0.645192,0.354645,0.407817]),
"surfboard": np.array([0.647385,0.217971,1.033455]),
"surveillance camera": np.array([0.242376,0.141928,0.271239]),
"sushi": np.array([0.479616,0.067626,0.232963]),
"swan": np.array([0.224991,0.056429,0.25539]),
"table tennis  paddle": np.array([1.153003,0.615771,0.271124]),
"tangerine": np.array([0.171045,0.054152,0.075107]),
"tape": np.array([0.149644,0.082444,0.076146]),
"tape measure/ruler": np.array([0.235281,0.086819,0.161372]),
"tea pot": np.array([0.303762,0.120477,0.160463]),
"telephone": np.array([0.39211,0.196959,0.136858]),
"tennis ball": np.array([0.381383,0.145496,0.0976]),
"tent": np.array([4.540731,0.529027,1.773167]),
"tie": np.array([0.395125,0.089519,0.291149]),
"tissue": np.array([0.222842,0.106449,0.109162]),
"toaster": np.array([0.502232,0.289022,0.240617]),
"toilet": np.array([0.705367,0.471736,0.621492]),
"toiletries": np.array([0.213948,0.098436,0.1224]),
"tomato": np.array([0.084943,0.054017,0.061882]),
"toothbrush": np.array([0.168137,0.036274,0.095008]),
"toothpaste": np.array([0.190628,0.074247,0.052137]),
"tortoise/turtle": np.array([0.327391,0.106473,0.136926]),
"towel/napkin": np.array([0.325807,0.170088,0.166185]),
"toy": np.array([0.294482,0.138423,0.125981]),
"traffic cone": np.array([0.510352,0.107394,0.398286]),
"trash bin/can": np.array([0.586903,0.31651,0.484136]),
"treadmill": np.array([1.09467,0.773162,1.0812]),
"tricycle": np.array([0.419115,0.289231,0.328601]),
"tripod": np.array([0.59516,0.278205,0.433376]),
"trolley": np.array([0.90536,0.606651,0.837939]),
"trophy": np.array([0.336382,0.112528,0.276777]),
"truck": np.array([0.588457,0.182021,0.210188]),
"umbrella": np.array([0.452864,0.293491,0.440433]),
"urinal": np.array([0.556061,0.371196,0.772323]),
"vent": np.array([0.328269,0.104269,0.134286]),
"violin": np.array([0.095734,0.022241,0.163189]),
"volleyball": np.array([0.320059,0.217041,0.254739]),
"washing machine": np.array([1.407989,0.448032,0.992492]),
"watch": np.array([0.113256,0.050778,0.077618]),
"wheelchair": np.array([2.178313,0.468531,0.797584]),
"wild bird": np.array([0.205464,0.051502,0.142949]),
"wine glass": np.array([0.326527,0.0905,0.201894]),
"zebra": np.array([0.882953,0.125619,0.487314]),
"sneakers": np.array([1,1,1]),
"street lights": np.array([1,1,1]),
"bracelet": np.array([1,1,1]),
"monitor": np.array([1,1,1]),
"suv": np.array([1,1,1]),
"traffic light": np.array([1,1,1]),
"van": np.array([1,1,1]),
"traffic sign": np.array([1,1,1]),
"motorbike/motorcycle": np.array([1,1,1]),
"paddle": np.array([1,1,1]),
"cow": np.array([1,1,1]),
"bus": np.array([1,1,1]),
"orange": np.array([1,1,1]),
"skis": np.array([1,1,1]),
"soccer": np.array([1,1,1]),
"combine with glove": np.array([1,1,1]),
"train": np.array([1,1,1]),
"machinery vehicle": np.array([1,1,1]),
"cleaning products": np.array([1,1,1]),
"sailboat": np.array([1,1,1]),
"gas stove": np.array([1,1,1]),
"shovel": np.array([1,1,1]),
"Life saver": np.array([1,1,1]),
"sports car": np.array([1,1,1]),
"converter": np.array([1,1,1]),
"carrot": np.array([1,1,1]),
"tennis racket": np.array([1,1,1]),
"baseball": np.array([1,1,1]),
"pigeon": np.array([1,1,1]),
"american football": np.array([1,1,1]),
"basketball": np.array([1,1,1]),
"billiards": np.array([1,1,1]),
"goose": np.array([1,1,1]),
"facial mask": np.array([1,1,1]),
"hamburger": np.array([1,1,1]),
"cucumber": np.array([1,1,1]),
"clutch": np.array([1,1,1]),
"tong": np.array([1,1,1]),
"slide": np.array([1,1,1]),
"hot dog": np.array([1,1,1]),
"deer": np.array([1,1,1]),
"ship": np.array([1,1,1]),
"chicken": np.array([1,1,1]),
"onion": np.array([1,1,1]),
"ice cream": np.array([1,1,1]),
"plum": np.array([1,1,1]),
"watermelon": np.array([1,1,1]),
"cabbage": np.array([1,1,1]),
"crane": np.array([1,1,1]),
"fire truck": np.array([1,1,1]),
"helicopter": np.array([1,1,1]),
"green beans": np.array([1,1,1]),
"carriage": np.array([1,1,1]),
"cigar": np.array([1,1,1]),
"hurdle": np.array([1,1,1]),
"swing": np.array([1,1,1]),
"french fries": np.array([1,1,1]),
"horn": np.array([1,1,1]),
"avocado": np.array([1,1,1]),
"saxophone": np.array([1,1,1]),
"trumpet": np.array([1,1,1]),
"cue": np.array([1,1,1]),
"bear": np.array([1,1,1]),
"tablet": np.array([1,1,1]),
"green vegetables": np.array([1,1,1]),
"nuts": np.array([1,1,1]),
"corn": np.array([1,1,1]),
"eggplant": np.array([1,1,1]),
"dates": np.array([1,1,1]),
"rice": np.array([1,1,1]),
"hamimelon": np.array([1,1,1]),
"lettuce": np.array([1,1,1]),
"meat balls": np.array([1,1,1]),
"medal": np.array([1,1,1]),
"shrimp": np.array([1,1,1]),
"rickshaw": np.array([1,1,1]),
"trombone": np.array([1,1,1]),
"jellyfish": np.array([1,1,1]),
"mushroom": np.array([1,1,1]),
"cheese": np.array([1,1,1]),
"race car": np.array([1,1,1]),
"tuba": np.array([1,1,1]),
"papaya": np.array([1,1,1]),
"green onion": np.array([1,1,1]),
"chips": np.array([1,1,1]),
"donkey": np.array([1,1,1]),
"spring rolls": np.array([1,1,1]),
"steak": np.array([1,1,1]),
"llama": np.array([1,1,1]),
"radish": np.array([1,1,1]),
"noodles": np.array([1,1,1]),
"yak": np.array([1,1,1]),
"crab": np.array([1,1,1]),
"baozi": np.array([1,1,1]),
"red cabbage": np.array([1,1,1]),
"polar bear": np.array([1,1,1]),
"seal": np.array([1,1,1]),
"pitaya": np.array([1,1,1]),
"okra": np.array([1,1,1]),
"durian": np.array([1,1,1]),
"rabbit": np.array([1,1,1]),
"ambulance": np.array([1,1,1]),
"asparagus": np.array([1,1,1]),
"pasta": np.array([1,1,1]),
"target": np.array([1,1,1]),
"chainsaw": np.array([1,1,1]),
"lobster": np.array([1,1,1]),}

        import pickle
        # with open('meansize_5k.pkl', 'rb') as file:
        #     self.type_mean_size = pickle.load(file)
        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]
        
        #testing set
        self.num_eval_class = 20 #change this 
        self.type2class_eval ={"chair": 0, "table": 1, "pillow": 2, "desk": 3, "bed": 4, "sofa": 5, "lamp": 6, "garbage_bin": 7, "cabinet": 8, "sink": 9, "night_stand": 10, "stool": 11, "bookshelf": 12, "dresser": 13, "toilet": 14, "fridge": 15, "microwave": 16, "counter": 17, "bathtub": 18, "scanner": 19}
        self.class2type_eval = {self.type2class_eval[t]:t for t in self.type2class_eval}
        #change this 
        self.type_mean_size_eval = {
                                'toilet' : np.array([0.697272, 0.454178, 0.75625]),
                                'bed' : np.array([2.115816, 1.621716, 0.936364]),
                                'chair' : np.array([0.592329, 0.552978, 0.827272]),
                                'bathtub' : np.array([0.767272, 1.398258, 0.45663]),
                                'sofa' : np.array([0.924369, 1.8750179999999999, 0.847046]),
                                'dresser' : np.array([0.528526, 1.0013420000000002, 1.183333]),
                                'scanner' : np.array([0.707552, 0.686108, 0.954546]),
                                'fridge' : np.array([0.732086, 0.7546, 1.65]),
                                'lamp' : np.array([0.367022, 0.379614, 0.69091]),
                                'desk' : np.array([0.68826, 1.337694, 0.7375]),
                                'table' : np.array([0.792666, 1.285808, 0.718182]),
                                'night_stand' : np.array([0.36723799999999995, 0.46594599999999997, 0.9227270000000001]),
                                'cabinet' : np.array([0.571631, 1.214407, 0.963636]),
                                'counter' : np.array([0.760644, 2.23633, 0.85]),
                                'garbage_bin' : np.array([0.297542, 0.356406, 0.75909]),
                                'bookshelf' : np.array([0.40298500000000004, 1.063498, 1.727273]),
                                'pillow' : np.array([0.355497, 0.56077, 0.318182]),
                                'microwave' : np.array([0.394077, 0.5639620000000001, 0.327272]),
                                'sink' : np.array([0.502248, 0.599351, 0.457344]),
                                'stool' : np.array([0.409797, 0.411632, 0.681818])}
        
        self.mean_size_arr_eval = np.zeros((self.num_eval_class, 3))
        for i in range(self.num_eval_class):
            self.mean_size_arr_eval[i,:] = self.type_mean_size_eval[self.class2type_eval[i]]
    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
    
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]    
        
        return size_class, size_residual
    def size2class_eval(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''

        size_class = self.type2class_eval[type_name]
        size_residual = size - self.type_mean_size_eval[type_name]
        
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]

        return mean_size + residual
    
    def class2size_eval(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size_eval[self.class2type_eval[pred_cls]]
        return mean_size + residual
    
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb
    
    def param2obb_eval(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size_eval(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb
    


