import os
import cv2
import json
import ujson
from PIL import Image
import open3d as o3d
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from math import gcd
import trimesh
import  keyboard
import scipy.io as sio
import threading
import multiprocessing as mp
type2class = {'door':0 ,'aerosol_can': 1, 'air_conditioner': 2, 'airplane': 3, 'alarm_clock': 4, 'alcohol': 5, 'alligator': 6, 'almond': 7, 'ambulance': 8, 'amplifier': 9, 'anklet': 10, 'antenna': 11, 'apple': 12, 'applesauce': 13, 'apricot': 14, 'apron': 15, 'aquarium': 16, 'arctic_(type_of_shoe)': 17, 'armband': 18, 'armchair': 19, 'armoire': 20, 'armor': 21, 'artichoke': 22, 'trash_can': 23, 'ashtray': 24, 'asparagus': 25, 'atomizer': 26, 'avocado': 27, 'award': 28, 'awning': 29, 'ax': 30, 'baboon': 31, 'baby_buggy': 32, 'basketball_backboard': 33, 'backpack': 34, 'handbag': 35, 'suitcase': 36, 'bagel': 37, 'bagpipe': 38, 'baguet': 39, 'bait': 40, 'ball': 41, 'ballet_skirt': 42, 'balloon': 43, 'bamboo': 44, 'banana': 45, 'Band_Aid': 46, 'bandage': 47, 'bandanna': 48, 'banjo': 49, 'banner': 50, 'barbell': 51, 'barge': 52, 'barrel': 53, 'barrette': 54, 'barrow': 55, 'baseball_base': 56, 'baseball': 57, 'baseball_bat': 58, 'baseball_cap': 59, 'baseball_glove': 60, 'basket': 61, 'basketball': 62, 'bass_horn': 63, 'bat_(animal)': 64, 'bath_mat': 65, 'bath_towel': 66, 'bathrobe': 67, 'bathtub': 68, 'batter_(food)': 69, 'battery': 70, 'beachball': 71, 'bead': 72, 'bean_curd': 73, 'beanbag': 74, 'beanie': 75, 'bear': 76, 'bed': 77, 'bedpan': 78, 'bedspread': 79, 'cow': 80, 'beef_(food)': 81, 'beeper': 82, 'beer_bottle': 83, 'beer_can': 84, 'beetle': 85, 'bell': 86, 'bell_pepper': 87, 'belt': 88, 'belt_buckle': 89, 'bench': 90, 'beret': 91, 'bib': 92, 'Bible': 93, 'bicycle': 94, 'visor': 95, 'billboard': 96, 'binder': 97, 'binoculars': 98, 'bird': 99, 'birdfeeder': 100, 'birdbath': 101, 'birdcage': 102, 'birdhouse': 103, 'birthday_cake': 104, 'birthday_card': 105, 'pirate_flag': 106, 'black_sheep': 107, 'blackberry': 108, 'blackboard': 109, 'blanket': 110, 'blazer': 111, 'blender': 112, 'blimp': 113, 'blinker': 114, 'blouse': 115, 'blueberry': 116, 'gameboard': 117, 'boat': 118, 'bob': 119, 'bobbin': 120, 'bobby_pin': 121, 'boiled_egg': 122, 'bolo_tie': 123, 'deadbolt': 124, 'bolt': 125, 'bonnet': 126, 'book': 127, 'bookcase': 128, 'booklet': 129, 'bookmark': 130, 'boom_microphone': 131, 'boot': 132, 'bottle': 133, 'bottle_opener': 134, 'bouquet': 135, 'bow_(weapon)': 136, 'bow_(decorative_ribbons)': 137, 'bow-tie': 138, 'bowl': 139, 'pipe_bowl': 140, 'bowler_hat': 141, 'bowling_ball': 142, 'box': 143, 'boxing_glove': 144, 'suspenders': 145, 'bracelet': 146, 'brass_plaque': 147, 'brassiere': 148, 'bread-bin': 149, 'bread': 150, 'breechcloth': 151, 'bridal_gown': 152, 'briefcase': 153, 'broccoli': 154, 'broach': 155, 'broom': 156, 'brownie': 157, 'brussels_sprouts': 158, 'bubble_gum': 159, 'bucket': 160, 'horse_buggy': 161, 'bull': 162, 'bulldog': 163, 'bulldozer': 164, 'bullet_train': 165, 'bulletin_board': 166, 'bulletproof_vest': 167, 'bullhorn': 168, 'bun': 169, 'bunk_bed': 170, 'buoy': 171, 'burrito': 172, 'bus_(vehicle)': 173, 'business_card': 174, 'butter': 175, 'butterfly': 176, 'button': 177, 'cab_(taxi)': 178, 'cabana': 179, 'cabin_car': 180, 'cabinet': 181, 'locker': 182, 'cake': 183, 'calculator': 184, 'calendar': 185, 'calf': 186, 'camcorder': 187, 'camel': 188, 'camera': 189, 'camera_lens': 190, 'camper_(vehicle)': 191, 'can': 192, 'can_opener': 193, 'candle': 194, 'candle_holder': 195, 'candy_bar': 196, 'candy_cane': 197, 'walking_cane': 198, 'canister': 199, 'canoe': 200, 'cantaloup': 201, 'canteen': 202, 'cap_(headwear)': 203, 'bottle_cap': 204, 'cape': 205, 'cappuccino': 206, 'car_(automobile)': 207, 'railcar_(part_of_a_train)': 208, 'elevator_car': 209, 'car_battery': 210, 'identity_card': 211, 'card': 212, 'cardigan': 213, 'cargo_ship': 214, 'carnation': 215, 'horse_carriage': 216, 'carrot': 217, 'tote_bag': 218, 'cart': 219, 'carton': 220, 'cash_register': 221, 'casserole': 222, 'cassette': 223, 'cast': 224, 'cat': 225, 'cauliflower': 226, 'cayenne_(spice)': 227, 'CD_player': 228, 'celery': 229, 'cellular_telephone': 230, 'chain_mail': 231, 'chair': 232, 'chaise_longue': 233, 'chalice': 234, 'chandelier': 235, 'chap': 236, 'checkbook': 237, 'checkerboard': 238, 'cherry': 239, 'chessboard': 240, 'chicken_(animal)': 241, 'chickpea': 242, 'chili_(vegetable)': 243, 'chime': 244, 'chinaware': 245, 'crisp_(potato_chip)': 246, 'poker_chip': 247, 'chocolate_bar': 248, 'chocolate_cake': 249, 'chocolate_milk': 250, 'chocolate_mousse': 251, 'choker': 252, 'chopping_board': 253, 'chopstick': 254, 'Christmas_tree': 255, 'slide': 256, 'cider': 257, 'cigar_box': 258, 'cigarette': 259, 'cigarette_case': 260, 'cistern': 261, 'clarinet': 262, 'clasp': 263, 'cleansing_agent': 264, 'cleat_(for_securing_rope)': 265, 'clementine': 266, 'clip': 267, 'clipboard': 268, 'clippers_(for_plants)': 269, 'cloak': 270, 'clock': 271, 'clock_tower': 272, 'clothes_hamper': 273, 'clothespin': 274, 'clutch_bag': 275, 'coaster': 276, 'coat': 277, 'coat_hanger': 278, 'coatrack': 279, 'cock': 280, 'cockroach': 281, 'cocoa_(beverage)': 282, 'coconut': 283, 'coffee_maker': 284, 'coffee_table': 285, 'coffeepot': 286, 'coil': 287, 'coin': 288, 'colander': 289, 'coleslaw': 290, 'coloring_material': 291, 'combination_lock': 292, 'pacifier': 293, 'comic_book': 294, 'compass': 295, 'computer_keyboard': 296, 'condiment': 297, 'cone': 298, 'control': 299, 'convertible_(automobile)': 300, 'sofa_bed': 301, 'cooker': 302, 'cookie': 303, 'cooking_utensil': 304, 'cooler_(for_food)': 305, 'cork_(bottle_plug)': 306, 'corkboard': 307, 'corkscrew': 308, 'edible_corn': 309, 'cornbread': 310, 'cornet': 311, 'cornice': 312, 'cornmeal': 313, 'corset': 314, 'costume': 315, 'cougar': 316, 'coverall': 317, 'cowbell': 318, 'cowboy_hat': 319, 'crab_(animal)': 320, 'crabmeat': 321, 'cracker': 322, 'crape': 323, 'crate': 324, 'crayon': 325, 'cream_pitcher': 326, 'crescent_roll': 327, 'crib': 328, 'crock_pot': 329, 'crossbar': 330, 'crouton': 331, 'crow': 332, 'crowbar': 333, 'crown': 334, 'crucifix': 335, 'cruise_ship': 336, 'police_cruiser': 337, 'crumb': 338, 'crutch': 339, 'cub_(animal)': 340, 'cube': 341, 'cucumber': 342, 'cufflink': 343, 'cup': 344, 'trophy_cup': 345, 'cupboard': 346, 'cupcake': 347, 'hair_curler': 348, 'curling_iron': 349, 'curtain': 350, 'cushion': 351, 'cylinder': 352, 'cymbal': 353, 'dagger': 354, 'dalmatian': 355, 'dartboard': 356, 'date_(fruit)': 357, 'deck_chair': 358, 'deer': 359, 'dental_floss': 360, 'desk': 361, 'detergent': 362, 'diaper': 363, 'diary': 364, 'die': 365, 'dinghy': 366, 'dining_table': 367, 'tux': 368, 'dish': 369, 'dish_antenna': 370, 'dishrag': 371, 'dishtowel': 372, 'dishwasher': 373, 'dishwasher_detergent': 374, 'dispenser': 375, 'diving_board': 376, 'Dixie_cup': 377, 'dog': 378, 'dog_collar': 379, 'doll': 380, 'dollar': 381, 'dollhouse': 382, 'dolphin': 383, 'domestic_ass': 384, 'doorknob': 385, 'doormat': 386, 'doughnut': 387, 'dove': 388, 'dragonfly': 389, 'drawer': 390, 'underdrawers': 391, 'dress': 392, 'dress_hat': 393, 'dress_suit': 394, 'dresser': 395, 'drill': 396, 'drone': 397, 'dropper': 398, 'drum_(musical_instrument)': 399, 'drumstick': 400, 'duck': 401, 'duckling': 402, 'duct_tape': 403, 'duffel_bag': 404, 'dumbbell': 405, 'dumpster': 406, 'dustpan': 407, 'eagle': 408, 'earphone': 409, 'earplug': 410, 'earring': 411, 'easel': 412, 'eclair': 413, 'eel': 414, 'egg': 415, 'egg_roll': 416, 'egg_yolk': 417, 'eggbeater': 418, 'eggplant': 419, 'electric_chair': 420, 'refrigerator': 421, 'elephant': 422, 'elk': 423, 'envelope': 424, 'eraser': 425, 'escargot': 426, 'eyepatch': 427, 'falcon': 428, 'fan': 429, 'faucet': 430, 'fedora': 431, 'ferret': 432, 'Ferris_wheel': 433, 'ferry': 434, 'fig_(fruit)': 435, 'fighter_jet': 436, 'figurine': 437, 'file_cabinet': 438, 'file_(tool)': 439, 'fire_alarm': 440, 'fire_engine': 441, 'fire_extinguisher': 442, 'fire_hose': 443, 'fireplace': 444, 'fireplug': 445, 'first-aid_kit': 446, 'fish': 447, 'fish_(food)': 448, 'fishbowl': 449, 'fishing_rod': 450, 'flag': 451, 'flagpole': 452, 'flamingo': 453, 'flannel': 454, 'flap': 455, 'flash': 456, 'flashlight': 457, 'fleece': 458, 'flip-flop_(sandal)': 459, 'flipper_(footwear)': 460, 'flower_arrangement': 461, 'flute_glass': 462, 'foal': 463, 'folding_chair': 464, 'food_processor': 465, 'football_(American)': 466, 'football_helmet': 467, 'footstool': 468, 'fork': 469, 'forklift': 470, 'freight_car': 471, 'French_toast': 472, 'freshener': 473, 'frisbee': 474, 'frog': 475, 'fruit_juice': 476, 'frying_pan': 477, 'fudge': 478, 'funnel': 479, 'futon': 480, 'gag': 481, 'garbage': 482, 'garbage_truck': 483, 'garden_hose': 484, 'gargle': 485, 'gargoyle': 486, 'garlic': 487, 'gasmask': 488, 'gazelle': 489, 'gelatin': 490, 'gemstone': 491, 'generator': 492, 'giant_panda': 493, 'gift_wrap': 494, 'ginger': 495, 'giraffe': 496, 'cincture': 497, 'glass_(drink_container)': 498, 'globe': 499, 'glove': 500, 'goat': 501, 'goggles': 502, 'goldfish': 503, 'golf_club': 504, 'golfcart': 505, 'gondola_(boat)': 506, 'goose': 507, 'gorilla': 508, 'gourd': 509, 'grape': 510, 'grater': 511, 'gravestone': 512, 'gravy_boat': 513, 'green_bean': 514, 'green_onion': 515, 'griddle': 516, 'grill': 517, 'grits': 518, 'grizzly': 519, 'grocery_bag': 520, 'guitar': 521, 'gull': 522, 'gun': 523, 'hairbrush': 524, 'hairnet': 525, 'hairpin': 526, 'halter_top': 527, 'ham': 528, 'hamburger': 529, 'hammer': 530, 'hammock': 531, 'hamper': 532, 'hamster': 533, 'hair_dryer': 534, 'hand_glass': 535, 'hand_towel': 536, 'handcart': 537, 'handcuff': 538, 'handkerchief': 539, 'handle': 540, 'handsaw': 541, 'hardback_book': 542, 'harmonium': 543, 'hat': 544, 'hatbox': 545, 'veil': 546, 'headband': 547, 'headboard': 548, 'headlight': 549, 'headscarf': 550, 'headset': 551, 'headstall_(for_horses)': 552, 'heart': 553, 'heater': 554, 'helicopter': 555, 'helmet': 556, 'heron': 557, 'highchair': 558, 'hinge': 559, 'hippopotamus': 560, 'hockey_stick': 561, 'hog': 562, 'home_plate_(baseball)': 563, 'honey': 564, 'fume_hood': 565, 'hook': 566, 'hookah': 567, 'hornet': 568, 'horse': 569, 'hose': 570, 'hot-air_balloon': 571, 'hotplate': 572, 'hot_sauce': 573, 'hourglass': 574, 'houseboat': 575, 'hummingbird': 576, 'hummus': 577, 'polar_bear': 578, 'icecream': 579, 'popsicle': 580, 'ice_maker': 581, 'ice_pack': 582, 'ice_skate': 583, 'igniter': 584, 'inhaler': 585, 'iPod': 586, 'iron_(for_clothing)': 587, 'ironing_board': 588, 'jacket': 589, 'jam': 590, 'jar': 591, 'jean': 592, 'jeep': 593, 'jelly_bean': 594, 'jersey': 595, 'jet_plane': 596, 'jewel': 597, 'jewelry': 598, 'joystick': 599, 'jumpsuit': 600, 'kayak': 601, 'keg': 602, 'kennel': 603, 'kettle': 604, 'key': 605, 'keycard': 606, 'kilt': 607, 'kimono': 608, 'kitchen_sink': 609, 'kitchen_table': 610, 'kite': 611, 'kitten': 612, 'kiwi_fruit': 613, 'knee_pad': 614, 'knife': 615, 'knitting_needle': 616, 'knob': 617, 'knocker_(on_a_door)': 618, 'koala': 619, 'lab_coat': 620, 'ladder': 621, 'ladle': 622, 'ladybug': 623, 'lamb_(animal)': 624, 'lamb-chop': 625, 'lamp': 626, 'lamppost': 627, 'lampshade': 628, 'lantern': 629, 'lanyard': 630, 'laptop_computer': 631, 'lasagna': 632, 'latch': 633, 'lawn_mower': 634, 'leather': 635, 'legging_(clothing)': 636, 'Lego': 637, 'legume': 638, 'lemon': 639, 'lemonade': 640, 'lettuce': 641, 'license_plate': 642, 'life_buoy': 643, 'life_jacket': 644, 'lightbulb': 645, 'lightning_rod': 646, 'lime': 647, 'limousine': 648, 'lion': 649, 'lip_balm': 650, 'liquor': 651, 'lizard': 652, 'log': 653, 'lollipop': 654, 'speaker_(stero_equipment)': 655, 'loveseat': 656, 'machine_gun': 657, 'magazine': 658, 'magnet': 659, 'mail_slot': 660, 'mailbox_(at_home)': 661, 'mallard': 662, 'mallet': 663, 'mammoth': 664, 'manatee': 665, 'mandarin_orange': 666, 'manger': 667, 'manhole': 668, 'map': 669, 'marker': 670, 'martini': 671, 'mascot': 672, 'mashed_potato': 673, 'masher': 674, 'mask': 675, 'mast': 676, 'mat_(gym_equipment)': 677, 'matchbox': 678, 'mattress': 679, 'measuring_cup': 680, 'measuring_stick': 681, 'meatball': 682, 'medicine': 683, 'melon': 684, 'microphone': 685, 'microscope': 686, 'microwave_oven': 687, 'milestone': 688, 'milk': 689, 'milk_can': 690, 'milkshake': 691, 'minivan': 692, 'mint_candy': 693, 'mirror': 694, 'mitten': 695, 'mixer_(kitchen_tool)': 696, 'money': 697, 'monitor_(computer_equipment)_computer_monitor': 698, 'monkey': 699, 'motor': 700, 'motor_scooter': 701, 'motor_vehicle': 702, 'motorcycle': 703, 'mound_(baseball)': 704, 'mouse_(computer_equipment)': 705, 'mousepad': 706, 'muffin': 707, 'mug': 708, 'mushroom': 709, 'music_stool': 710, 'musical_instrument': 711, 'nailfile': 712, 'napkin': 713, 'neckerchief': 714, 'necklace': 715, 'necktie': 716, 'needle': 717, 'nest': 718, 'newspaper': 719, 'newsstand': 720, 'nightshirt': 721, 'nosebag_(for_animals)': 722, 'noseband_(for_animals)': 723, 'notebook': 724, 'notepad': 725, 'nut': 726, 'nutcracker': 727, 'oar': 728, 'octopus_(food)': 729, 'octopus_(animal)': 730, 'oil_lamp': 731, 'olive_oil': 732, 'omelet': 733, 'onion': 734, 'orange_(fruit)': 735, 'orange_juice': 736, 'ostrich': 737, 'ottoman': 738, 'oven': 739, 'overalls_(clothing)': 740, 'owl': 741, 'packet': 742, 'inkpad': 743, 'pad': 744, 'paddle': 745, 'padlock': 746, 'paintbrush': 747, 'painting': 748, 'pajamas': 749, 'palette': 750, 'pan_(for_cooking)': 751, 'pan_(metal_container)': 752, 'pancake': 753, 'pantyhose': 754, 'papaya': 755, 'paper_plate': 756, 'paper_towel': 757, 'paperback_book': 758, 'paperweight': 759, 'parachute': 760, 'parakeet': 761, 'parasail_(sports)': 762, 'parasol': 763, 'parchment': 764, 'parka': 765, 'parking_meter': 766, 'parrot': 767, 'passenger_car_(part_of_a_train)': 768, 'passenger_ship': 769, 'passport': 770, 'pastry': 771, 'patty_(food)': 772, 'pea_(food)': 773, 'peach': 774, 'peanut_butter': 775, 'pear': 776, 'peeler_(tool_for_fruit_and_vegetables)': 777, 'wooden_leg': 778, 'pegboard': 779, 'pelican': 780, 'pen': 781, 'pencil': 782, 'pencil_box': 783, 'pencil_sharpener': 784, 'pendulum': 785, 'penguin': 786, 'pennant': 787, 'penny_(coin)': 788, 'pepper': 789, 'pepper_mill': 790, 'perfume': 791, 'persimmon': 792, 'person': 793, 'pet': 794, 'pew_(church_bench)': 795, 'phonebook': 796, 'phonograph_record': 797, 'piano': 798, 'pickle': 799, 'pickup_truck': 800, 'pie': 801, 'pigeon': 802, 'piggy_bank': 803, 'pillow': 804, 'pin_(non_jewelry)': 805, 'pineapple': 806, 'pinecone': 807, 'ping-pong_ball': 808, 'pinwheel': 809, 'tobacco_pipe': 810, 'pipe': 811, 'pistol': 812, 'pita_(bread)': 813, 'pitcher_(vessel_for_liquid)': 814, 'pitchfork': 815, 'pizza': 816, 'place_mat': 817, 'plate': 818, 'platter': 819, 'playpen': 820, 'pliers': 821, 'plow_(farm_equipment)': 822, 'plume': 823, 'pocket_watch': 824, 'pocketknife': 825, 'poker_(fire_stirring_tool)': 826, 'pole': 827, 'polo_shirt': 828, 'poncho': 829, 'pony': 830, 'pool_table': 831, 'pop_(soda)': 832, 'postbox_(public)': 833, 'postcard': 834, 'poster': 835, 'pot': 836, 'flowerpot': 837, 'potato': 838, 'potholder': 839, 'pottery': 840, 'pouch': 841, 'power_shovel': 842, 'prawn': 843, 'pretzel': 844, 'printer': 845, 'projectile_(weapon)': 846, 'projector': 847, 'propeller': 848, 'prune': 849, 'pudding': 850, 'puffer_(fish)': 851, 'puffin': 852, 'pug-dog': 853, 'pumpkin': 854, 'puncher': 855, 'puppet': 856, 'puppy': 857, 'quesadilla': 858, 'quiche': 859, 'quilt': 860, 'rabbit': 861, 'race_car': 862, 'racket': 863, 'radar': 864, 'radiator': 865, 'radio_receiver': 866, 'radish': 867, 'raft': 868, 'rag_doll': 869, 'raincoat': 870, 'ram_(animal)': 871, 'raspberry': 872, 'rat': 873, 'razorblade': 874, 'reamer_(juicer)': 875, 'rearview_mirror': 876, 'receipt': 877, 'recliner': 878, 'record_player': 879, 'reflector': 880, 'remote_control': 881, 'rhinoceros': 882, 'rib_(food)': 883, 'rifle': 884, 'ring': 885, 'river_boat': 886, 'road_map': 887, 'robe': 888, 'rocking_chair': 889, 'rodent': 890, 'roller_skate': 891, 'Rollerblade': 892, 'rolling_pin': 893, 'root_beer': 894, 'router_(computer_equipment)': 895, 'rubber_band': 896, 'runner_(carpet)': 897, 'plastic_bag': 898, 'saddle_(on_an_animal)': 899, 'saddle_blanket': 900, 'saddlebag': 901, 'safety_pin': 902, 'sail': 903, 'salad': 904, 'salad_plate': 905, 'salami': 906, 'salmon_(fish)': 907, 'salmon_(food)': 908, 'salsa': 909, 'saltshaker': 910, 'sandal_(type_of_shoe)': 911, 'sandwich': 912, 'satchel': 913, 'saucepan': 914, 'saucer': 915, 'sausage': 916, 'sawhorse': 917, 'saxophone': 918, 'scale_(measuring_instrument)': 919, 'scarecrow': 920, 'scarf': 921, 'school_bus': 922, 'scissors': 923, 'scoreboard': 924, 'scraper': 925, 'screwdriver': 926, 'scrubbing_brush': 927, 'sculpture': 928, 'seabird': 929, 'seahorse': 930, 'seaplane': 931, 'seashell': 932, 'sewing_machine': 933, 'shaker': 934, 'shampoo': 935, 'shark': 936, 'sharpener': 937, 'Sharpie': 938, 'shaver_(electric)': 939, 'shaving_cream': 940, 'shawl': 941, 'shears': 942, 'sheep': 943, 'shepherd_dog': 944, 'sherbert': 945, 'shield': 946, 'shirt': 947, 'shoe': 948, 'shopping_bag': 949, 'shopping_cart': 950, 'short_pants': 951, 'shot_glass': 952, 'shoulder_bag': 953, 'shovel': 954, 'shower_head': 955, 'shower_cap': 956, 'shower_curtain': 957, 'shredder_(for_paper)': 958, 'signboard': 959, 'silo': 960, 'sink': 961, 'skateboard': 962, 'skewer': 963, 'ski': 964, 'ski_boot': 965, 'ski_parka': 966, 'ski_pole': 967, 'skirt': 968, 'skullcap': 969, 'sled': 970, 'sleeping_bag': 971, 'sling_(bandage)': 972, 'slipper_(footwear)': 973, 'smoothie': 974, 'snake': 975, 'snowboard': 976, 'snowman': 977, 'snowmobile': 978, 'soap': 979, 'soccer_ball': 980, 'sock': 981, 'sofa': 982, 'softball': 983, 'solar_array': 984, 'sombrero': 985, 'soup': 986, 'soup_bowl': 987, 'soupspoon': 988, 'sour_cream': 989, 'soya_milk': 990, 'space_shuttle': 991, 'sparkler_(fireworks)': 992, 'spatula': 993, 'spear': 994, 'spectacles': 995, 'spice_rack': 996, 'spider': 997, 'crawfish': 998, 'sponge': 999, 'spoon': 1000, 'sportswear': 1001, 'spotlight': 1002, 'squid_(food)': 1003, 'squirrel': 1004, 'stagecoach': 1005, 'stapler_(stapling_machine)': 1006, 'starfish': 1007, 'statue_(sculpture)': 1008, 'steak_(food)': 1009, 'steak_knife': 1010, 'steering_wheel': 1011, 'stepladder': 1012, 'step_stool': 1013, 'stereo_(sound_system)': 1014, 'stew': 1015, 'stirrer': 1016, 'stirrup': 1017, 'stool': 1018, 'stop_sign': 1019, 'brake_light': 1020, 'stove': 1021, 'strainer': 1022, 'strap': 1023, 'straw_(for_drinking)': 1024, 'strawberry': 1025, 'street_sign': 1026, 'streetlight': 1027, 'string_cheese': 1028, 'stylus': 1029, 'subwoofer': 1030, 'sugar_bowl': 1031, 'sugarcane_(plant)': 1032, 'suit_(clothing)': 1033, 'sunflower': 1034, 'sunglasses': 1035, 'sunhat': 1036, 'surfboard': 1037, 'sushi': 1038, 'mop': 1039, 'sweat_pants': 1040, 'sweatband': 1041, 'sweater': 1042, 'sweatshirt': 1043, 'sweet_potato': 1044, 'swimsuit': 1045, 'sword': 1046, 'syringe': 1047, 'Tabasco_sauce': 1048, 'table-tennis_table': 1049, 'table': 1050, 'table_lamp': 1051, 'tablecloth': 1052, 'tachometer': 1053, 'taco': 1054, 'tag': 1055, 'taillight': 1056, 'tambourine': 1057, 'army_tank': 1058, 'tank_(storage_vessel)': 1059, 'tank_top_(clothing)': 1060, 'tape_(sticky_cloth_or_paper)': 1061, 'tape_measure': 1062, 'tapestry': 1063, 'tarp': 1064, 'tartan': 1065, 'tassel': 1066, 'tea_bag': 1067, 'teacup': 1068, 'teakettle': 1069, 'teapot': 1070, 'teddy_bear': 1071, 'telephone': 1072, 'telephone_booth': 1073, 'telephone_pole': 1074, 'telephoto_lens': 1075, 'television_camera': 1076, 'television_set': 1077, 'tennis_ball': 1078, 'tennis_racket': 1079, 'tequila': 1080, 'thermometer': 1081, 'thermos_bottle': 1082, 'thermostat': 1083, 'thimble': 1084, 'thread': 1085, 'thumbtack': 1086, 'tiara': 1087, 'tiger': 1088, 'tights_(clothing)': 1089, 'timer': 1090, 'tinfoil': 1091, 'tinsel': 1092, 'tissue_paper': 1093, 'toast_(food)': 1094, 'toaster': 1095, 'toaster_oven': 1096, 'toilet': 1097, 'toilet_tissue': 1098, 'tomato': 1099, 'tongs': 1100, 'toolbox': 1101, 'toothbrush': 1102, 'toothpaste': 1103, 'toothpick': 1104, 'cover': 1105, 'tortilla': 1106, 'tow_truck': 1107, 'towel': 1108, 'towel_rack': 1109, 'toy': 1110, 'tractor_(farm_equipment)': 1111, 'traffic_light': 1112, 'dirt_bike': 1113, 'trailer_truck': 1114, 'train_(railroad_vehicle)': 1115, 'trampoline': 1116, 'tray': 1117, 'trench_coat': 1118, 'triangle_(musical_instrument)': 1119, 'tricycle': 1120, 'tripod': 1121, 'trousers': 1122, 'truck': 1123, 'truffle_(chocolate)': 1124, 'trunk': 1125, 'vat': 1126, 'turban': 1127, 'turkey_(food)': 1128, 'turnip': 1129, 'turtle': 1130, 'turtleneck_(clothing)': 1131, 'typewriter': 1132, 'umbrella': 1133, 'underwear': 1134, 'unicycle': 1135, 'urinal': 1136, 'urn': 1137, 'vacuum_cleaner': 1138, 'vase': 1139, 'vending_machine': 1140, 'vent': 1141, 'vest': 1142, 'videotape': 1143, 'vinegar': 1144, 'violin': 1145, 'vodka': 1146, 'volleyball': 1147, 'vulture': 1148, 'waffle': 1149, 'waffle_iron': 1150, 'wagon': 1151, 'wagon_wheel': 1152, 'walking_stick': 1153, 'wall_clock': 1154, 'wall_socket': 1155, 'wallet': 1156, 'walrus': 1157, 'wardrobe': 1158, 'washbasin': 1159, 'automatic_washer': 1160, 'watch': 1161, 'water_bottle': 1162, 'water_cooler': 1163, 'water_faucet': 1164, 'water_heater': 1165, 'water_jug': 1166, 'water_gun': 1167, 'water_scooter': 1168, 'water_ski': 1169, 'water_tower': 1170, 'watering_can': 1171, 'watermelon': 1172, 'weathervane': 1173, 'webcam': 1174, 'wedding_cake': 1175, 'wedding_ring': 1176, 'wet_suit': 1177, 'wheel': 1178, 'wheelchair': 1179, 'whipped_cream': 1180, 'whistle': 1181, 'wig': 1182, 'wind_chime': 1183, 'windmill': 1184, 'window_box_(for_plants)': 1185, 'windshield_wiper': 1186, 'windsock': 1187, 'wine_bottle': 1188, 'wine_bucket': 1189, 'wineglass': 1190, 'blinder_(for_horses)': 1191, 'wok': 1192, 'wolf': 1193, 'wooden_spoon': 1194, 'wreath': 1195, 'wrench': 1196, 'wristband': 1197, 'wristlet': 1198, 'yacht': 1199, 'yogurt': 1200, 'yoke_(animal_equipment)': 1201, 'zebra': 1202, 'zucchini': 1203}

def get_intrinsics(H, W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)

    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def depth_to_points(depth, R=None,K=None, t=None):
    #if K is None:
    #K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    angle_x = np.radians(-90)
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    R = R@Rx
    #R =Rx

    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    scales = 8000
    #print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D/scales * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    #pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]

def depth_edges_mask(depth,thr):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > thr  # 0.01
    return mask


def normalize_depth(depth, min_depth=None, max_depth=None):

    if min_depth is None:
        min_depth = np.min(depth)
    if max_depth is None:
        max_depth = np.max(depth)

    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth

def get_pointcloud_and_bbox(image, depth,Rtilt,K,thr =0.05,remove_edges=True):
    #image.thumbnail((1024, 1024))  # limit the size of the input image
    depth = np.array(depth).astype(np.uint16)
    #depth =  np.max(depth) - depth
   #depth =  0.5 + normalize_depth(depth)

    if remove_edges:
        # Compute the edge mask.
        depth_remove =normalize_depth(depth)
        mask = depth_edges_mask(depth_remove,thr)
        # Filter the depth map using the edge mask.
        depth[mask] = 0.0

    pts3d = depth_to_points(depth[None],R=Rtilt,K = K)
    pts3d = pts3d.reshape(-1, 3)

    # Get RGB image
    rgb = np.array(image.convert('RGB'))
    # Convert to Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)

    #sunrgbd_format.
    y_values = pts3d[:, 1]  # 提取y轴的值
    nonzero_indices = np.nonzero(y_values)  # 找到非零y值的索引
    pcd = pcd.select_by_index(nonzero_indices[0])  # 通过索引删除对应的点

    # 获取 RGB 通道
    colors = np.asarray(pcd.colors)
    # 获取黑色点的索引
    black_indices = np.where(np.all(colors == [0, 0, 0], axis=1))[0]
    # 删除黑色点
    pcd_without_black = pcd.select_by_index(np.delete(np.arange(len(colors)), black_indices))
    # Save as ply
    return pcd_without_black

def render_to_2D(pc,Rtilt,K,H,W):
    angle_x = np.radians(90)
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    Rtilt_inverse = np.linalg.inv(Rtilt)
    xyz2 = np.dot((Rtilt).transpose(), pc.transpose())
    xyz2 = xyz2.transpose()
    xyz2[:,[0,1,2]] = xyz2[:,[0,2,1]]
    xyz2[:,1] *= -1
    uv = np.dot(xyz2, K.transpose())
    uv[:,0] /= uv[:,2]
    uv[:,1] /= uv[:,2]
    # h, w = int(np.max(uv[:, 1])) + 1, int(np.max(uv[:, 0])) + 1
    
    # print(h,w)
    # depth_image = np.zeros((H,W))
    # depth = pc[:,2]

    # for i in range(uv.shape[0]):
    #     u, v = int(uv[i, 0]), int(uv[i, 1])
    
        # depth_image[v, u] = depth[i]
    
    # depth_image = (depth_image * 255).astype(np.uint8)  # Convert to 8-bit
    # depth_image = Image.fromarray(depth_image)#.transpose(Image.FLIP_LEFT_RIGHT)
    # depth_image.save('depth_image_eval.png')

    # Clip to image boundaries
    uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
    
    # Compute bounding box in COCO format
    x_min, y_min = np.min(uv[:, 0]), np.min(uv[:, 1])
    x_max, y_max = np.max(uv[:, 0]), np.max(uv[:, 1])
    bbox_width, bbox_height = x_max - x_min, y_max - y_min
    
    # Return bounding box in COCO format
    bbox_coco_format = [x_min, y_min, bbox_width, bbox_height]
    return bbox_coco_format


def lift_and_render(rgb_filenames, depth_filenames, calib_filenames,gt_image_calib_filenames,controlnet_filenames,rgb_folder, depth_folder,controlnet_folder,calib_folder ,gt_image_calib_folder,instance_json_path, output_json, output_image_folder, annotations):
    data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    for rgb_filename, depth_filename, calib_filename,controlnet_filename,gt_image_calib_filename in zip(rgb_filenames, depth_filenames,calib_filenames,controlnet_filenames,gt_image_calib_filenames):
        if (rgb_filename.split('.')[0] != depth_filename.split('.')[0]) or (rgb_filename.split('.')[0] != calib_filename.split('.')[0] != controlnet_filename.split('.')[0] != gt_image_calib_filename.split('.')[0] ):
            print("rgb_filename, depth_filename", rgb_filename, depth_filename,calib_filename)
            raise ValueError("文件名不相同，终止运行。")
        rgb_filepath = os.path.join(rgb_folder, rgb_filename)
        depth_filepath = os.path.join(depth_folder, depth_filename)
        calib_filepath = os.path.join(calib_folder, calib_filename)
        controlnet_filepath = os.path.join(controlnet_folder, controlnet_filename)
        gt_image_calib_filepath = os.path.join(gt_image_calib_folder, gt_image_calib_filename)
        
        print("controlnet_filepath",controlnet_filepath)
        img_controllnet = cv2.imread(controlnet_filepath)
        H_control,W_control,_ = img_controllnet.shape
        image_id_control = controlnet_filename.split('.')[0]
        data['images'].append({
                    'file_name': f'{image_id_control}.png',
                    'height': H_control,
                    'width': W_control,
                    'id': int(image_id_control)
                })
        image = Image.open(rgb_filepath)
        image = image.convert('RGB')
        if depth_filepath.endswith('.pfm'):
            # 读取PFM格式深度图像
            depth = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
            #print("depth_filepath", depth_filepath)
            # 将深度图像转换为PIL Image对象

            depth = Image.fromarray(depth)
        elif depth_filepath.endswith('.png'):
            # 读取PNG格式深度图像
            depth = Image.open(depth_filepath)
        else:
            raise ValueError('Unknown file format: {}'.format(depth_filepath))

        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        Rtilt = np.reshape(Rtilt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        K = np.reshape(K, (3, 3), order='F')
        print("R",Rtilt)
        print("K",K)

        lines = [line.rstrip() for line in open(gt_image_calib_filepath)]
        gt_Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        gt_Rtilt = np.reshape(gt_Rtilt, (3, 3), order='F')
        gt_K = np.array([float(x) for x in lines[1].split(' ')])
        gt_K = np.reshape(gt_K, (3, 3), order='F')
        print("gt_R",gt_Rtilt)
        print("gt_K",gt_K)


        num_str = rgb_filepath.split("/")[-1].split(".")[0]
        # Create empty mask
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Create empty masks arrayget_pointcloud
        H,W = image.shape[0], image.shape[1]
        image_anns = [ann for ann in annotations['annotations'] if ann['image_id'] == int(num_str)]
        masks = np.zeros((len(image_anns), H, W), np.uint8)
        classes=[]
        bounding_boxes2d = []

        '''
        for i, ann in enumerate(image_anns):
            if isinstance(ann['segmentation'], dict):
                continue
            bbox2d = ann['bbox']
            bounding_boxes2d.append(bbox2d)
            seg = ann['segmentation'][0]
            classes.append(ann['category_id'])
            pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(masks[i,:,:],[pts], 255)
        '''
        for i, ann in enumerate(image_anns):
            bbox2d = ann['bbox']
            bounding_boxes2d.append(bbox2d)
            classes.append(ann['category_id'])
            # If your segmentation data contains multiple lists,
            # iterate over each list (polygon) in the segmentation data
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                # The fillPoly function is called for each polygon
                cv2.fillPoly(masks[i,:,:],[pts], 255)
        
        print("calsses",classes)
        instances=[]
        instances_sample=[]
        instances_DBSCAN = []
        bboxs=[]
        valid_ins=[]
        sunrgbdformat_bboxs = []
        DBSCAN_flag=True
        for i, mask in enumerate(masks):
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = Image.fromarray(mask)
            pcd = get_pointcloud_and_bbox(mask, depth,gt_Rtilt,gt_K,thr=0.005)
            instances.append(pcd)
            #
            #去掉一模一样的干扰点
            _, unique_indices = np.unique(pcd.points, axis=0, return_index=True)
            pc_for_bbox = np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)])[unique_indices]
            if DBSCAN_flag:
                step_interval = max((1, int(pc_for_bbox.shape[0] / 3000)))
                cur_ins_pc = pc_for_bbox[0:pc_for_bbox.shape[0]:step_interval, :]

                if cur_ins_pc.shape[0] < 100:
                    print("loss")
                    continue
                instances_sample.append(cur_ins_pc)

                db = DBSCAN(eps=0.3, min_samples=100).fit(cur_ins_pc)

                cur_ins_pc_remove_outiler = []
                for cluster in np.unique(db.labels_):
                    if cluster < 0:
                        continue

                    cluster_ind = np.where(db.labels_ == cluster)[0]

                    if cluster_ind.shape[0] / cur_ins_pc.shape[0] < 0.2 or cluster_ind.shape[0] <= 100:
                        continue
                    cur_ins_pc_remove_outiler.append(cur_ins_pc[cluster_ind, :])

       


                if len(cur_ins_pc_remove_outiler) < 1:
                    continue

                valid_ins.append(i)

                pc_for_bbox = np.concatenate(cur_ins_pc_remove_outiler, axis=0)
                print(pc_for_bbox.shape,rgb_filename,H_control,W_control)
                bbox_coco_format = render_to_2D(pc_for_bbox[:,:3],Rtilt,K,H_control,W_control)
                area = bbox_coco_format[2]*bbox_coco_format[3]
                data['annotations'].append({
                    'id': len(data['annotations']) + 1,
                    'image_id': int(image_id_control),
                    'category_id': classes[i],
                    'bbox': bbox_coco_format,
                    'area': area,
                    'iscrowd': 0
                })
                instances_DBSCAN.append(pc_for_bbox)
    return data



import argparse
# 导入argparse库，用于解析命令行参数
parser = argparse.ArgumentParser(description='Process files for a specific task')
parser.add_argument('--num_tasks',default =1, type=int, help='Total number of tasks')
parser.add_argument('--task_number',default =1, type=int, help='Task number (starting from 1)')
args = parser.parse_args()
world_size = 16

def worker_task(args):
    # Unpack arguments
    
    idx, shared_data, lock, rgb_filenames, depth_filenames, calib_filenames,gt_image_calib_filenames, controlnet_filenames, rgb_folder, depth_folder, controlnet_folder, calib_folder,gt_image_calib_folder, instance_json_path, output_json, output_image_folder, annotations = args
    
    # Partial data processing by this worker
    local_data = lift_and_render(
        rgb_filenames[idx::world_size], depth_filenames[idx::world_size],
        calib_filenames[idx::world_size],gt_image_calib_filenames[idx::world_size], controlnet_filenames[idx::world_size],
        rgb_folder, depth_folder, controlnet_folder, calib_folder,gt_image_calib_folder,
        instance_json_path, output_json, output_image_folder, annotations
    )
    
    # Lock to update shared data safely
    with lock:
        # 假设我们有一些局部数据要添加
        temp_images = shared_data['images']
        temp_images.extend(local_data['images'])
        shared_data['images'] = temp_images  # 显式重新赋值

        temp_annotations = shared_data['annotations']
        temp_annotations.extend(local_data['annotations'])
        shared_data['annotations'] = temp_annotations

        temp_categories = shared_data['categories']
        temp_categories.extend(local_data['categories'])
        shared_data['categories'] = temp_categories

def process(rgb_folder, depth_folder, controlnet_folder,calib_folder,gt_image_calib_folder,instance_json_path, output_json, output_image_folder):
    
    with open('/share/timingyang/Detic_pseudo_finetune/3Dboxrender2D_ad/id_list.txt', 'r') as file:
        file_ids = [line.strip() for line in file]

    # 定义任务分割参数
    num_tasks = args.num_tasks  # 总共分成10份任务
    task_number = args.task_number  # 指定第几份任务，从1开始

    # 计算每份任务的文件ID范围
    total_files = len(file_ids)
    print("len(file_ids)",len(file_ids))
    files_per_task = total_files // num_tasks
    start_idx = (task_number - 1) * files_per_task
    end_idx = start_idx + files_per_task
    # 如果有余数，增加end_idx以包括多余的文件ID
    if task_number == num_tasks:
        end_idx = total_files

    # 获取当前任务的文件ID子列表
    task_file_ids = file_ids[start_idx:end_idx]
    print(task_file_ids[:2],task_file_ids[-2:])
    print("task_file_ids",len(task_file_ids))
    # 初始化空列表以存储文件名
    rgb_filenames = []
    depth_filenames = []
    calib_filenames = []
    gt_image_calib_filenames = []
    controlnet_filenames = []
    # 构建文件名并添加到列表中
    for file_id in task_file_ids:
        file_id = str(file_id).zfill(6)
        rgb_filenames.append(f'{file_id}.jpg')
        depth_filenames.append(f'{file_id}.png')
        calib_filenames.append(f'{file_id}.txt')
        gt_image_calib_filenames.append(f'{file_id}.txt')
        controlnet_filenames.append(f'{file_id}.png')
    
    with open(instance_json_path, "r") as f:
        annotations = json.load(f)

    
    with mp.Manager() as manager:
        # 创建共享字典
        shared_data = manager.dict({
            'images': [],
            'annotations': [],
            'categories': type2class
        })
        data_cate = [{'id': v, 'name': k} for k, v in type2class.items()]
        shared_data['categories'] = data_cate
        lock = manager.Lock()

        # Define the number of processes
        

        # Prepare arguments for each process
        process_args = [
            (
                i, shared_data, lock, rgb_filenames, depth_filenames,
                calib_filenames,gt_image_calib_filenames, controlnet_filenames, rgb_folder, depth_folder,
                controlnet_folder, calib_folder,gt_image_calib_folder, instance_json_path,
                output_json, output_image_folder, annotations
            )
            for i in range(world_size)
        ]

        # Create a pool of processes
        with mp.Pool(processes=world_size) as pool:
            pool.map(worker_task, process_args)

        # Convert shared_data back to a regular dictionary for further use
        data = dict(shared_data)

        # Assuming the output_json is a path where you want to save the JSON
        with open(output_json, 'w') as f:
            ujson.dump(data, f)




# def process(rgb_folder, depth_folder, controlnet_folder,calib_folder,gt_image_calib_folder,instance_json_path, output_json, output_image_folder):
    
#     with open('/share/timingyang/Detic_pseudo_finetune/3Dboxrender2D_ad/id_list.txt', 'r') as file:
#         file_ids = [line.strip() for line in file]

#     # 定义任务分割参数
#     num_tasks = args.num_tasks  # 总共分成10份任务
#     task_number = args.task_number  # 指定第几份任务，从1开始

#     # 计算每份任务的文件ID范围
#     total_files = len(file_ids)
#     print("len(file_ids)",len(file_ids))
#     files_per_task = total_files // num_tasks
#     start_idx = (task_number - 1) * files_per_task
#     end_idx = start_idx + files_per_task
#     # 如果有余数，增加end_idx以包括多余的文件ID
#     if task_number == num_tasks:
#         end_idx = total_files

#     # 获取当前任务的文件ID子列表
#     task_file_ids = file_ids[start_idx:end_idx]
#     print(task_file_ids[:2],task_file_ids[-2:])
#     print("task_file_ids",len(task_file_ids))
#     # 初始化空列表以存储文件名
#     rgb_filenames = []
#     depth_filenames = []
#     calib_filenames = []
#     gt_image_calib_filenames =[]
#     controlnet_filenames = []
#     # 构建文件名并添加到列表中
#     for file_id in task_file_ids:
#         file_id = str(file_id).zfill(6)
#         rgb_filenames.append(f'{file_id}.jpg')
#         depth_filenames.append(f'{file_id}.png')
#         calib_filenames.append(f'{file_id}.txt')
#         gt_image_calib_filenames.append(f'{file_id}.txt')
#         controlnet_filenames.append(f'{file_id}.png')
    
#     with open(instance_json_path, "r") as f:
#         annotations = json.load(f)

    
#     data = lift_and_render(rgb_filenames, depth_filenames, calib_filenames,gt_image_calib_filenames,controlnet_filenames,rgb_folder, depth_folder,controlnet_folder,calib_folder,gt_image_calib_folder ,instance_json_path, output_json, output_image_folder, annotations)
#     output_file_path = os.path.join(output_json_folder, "json_output_new.json")
#     with open(output_file_path, 'w') as f:
#         ujson.dump(dict(data), f)



if __name__ == "__main__":
    
    '''
    step1: lift GT 2D data to 3D space using GT calib
    step2: render to 2D data[bbox for finetune detic] using calib from Data_Maker/PointCloudRender/Finetune_data_creater/inference_data_creater.py
    '''
    
    #GT image
    rgb_folder = "/share/timingyang/baidu/sunrgbd_trainval/image"
    #GT Calib
    gt_image_calib_folder = '/share/timingyang/baidu/sunrgbd_trainval/calib'
    #GT depth
    depth_folder = '/share/timingyang/baidu/depth_image'
    #GT/Detic output of instance[coco format]
    instance_json_path = '/share/timingyang/Detic_pseudo_finetune/3Dboxrender2D_1204_sunrgbd/json_gt_image.json'
    #pseudo image from controlnet
    controlnet_folder = '/share1/timingyang/20240216/AD_WANDB/SUNRGBD/2DBARNCH_365_human/10percent/imvotenet_wandb_indoorimage_pretrain_nofixed/sunrgbd/sunrgbd_trainval_eval/image'
    #calib from Data_Maker/PointCloudRender/Finetune_data_creater/inference_data_creater.py
    calib_folder = '/share1/timingyang/20240216/AD_WANDB/SUNRGBD/2DBARNCH_365_human/10percent/imvotenet_wandb_indoorimage_pretrain_nofixed/sunrgbd/sunrgbd_trainval_eval/calib'
    
    #output json for Finetune Detic
    output_json = 'output.json'
    
    #useless
    output_image_folder = './outimage'

    #Path(output_json).mkdir(parents=True, exist_ok=True)
    Path(output_image_folder).mkdir(parents=True, exist_ok=True)
    process(rgb_folder, depth_folder, controlnet_folder,calib_folder,gt_image_calib_folder,instance_json_path, output_json, output_image_folder)