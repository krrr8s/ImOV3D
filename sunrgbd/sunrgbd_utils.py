# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Provides Python helper function to read My SUNRGBD dataset.

Author: Charles R. Qi
Date: October, 2017

Updated by Charles R. Qi
Date: December, 2018
Note: removed basis loading.
'''
import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points

type2class={'door':0, 'aerosol_can': 1, 'air_conditioner': 2, 'airplane': 3, 'alarm_clock': 4, 'alcohol': 5, 'alligator': 6, 'almond': 7, 'ambulance': 8, 'amplifier': 9, 'anklet': 10, 'antenna': 11, 'apple': 12, 'applesauce': 13, 'apricot': 14, 'apron': 15, 'aquarium': 16, 'arctic_(type_of_shoe)': 17, 'armband': 18, 'armchair': 19, 'armoire': 20, 'armor': 21, 'artichoke': 22, 'trash_can': 23, 'ashtray': 24, 'asparagus': 25, 'atomizer': 26, 'avocado': 27, 'award': 28, 'awning': 29, 'ax': 30, 'baboon': 31, 'baby_buggy': 32, 'basketball_backboard': 33, 'backpack': 34, 'handbag': 35, 'suitcase': 36, 'bagel': 37, 'bagpipe': 38, 'baguet': 39, 'bait': 40, 'ball': 41, 'ballet_skirt': 42, 'balloon': 43, 'bamboo': 44, 'banana': 45, 'Band_Aid': 46, 'bandage': 47, 'bandanna': 48, 'banjo': 49, 'banner': 50, 'barbell': 51, 'barge': 52, 'barrel': 53, 'barrette': 54, 'barrow': 55, 'baseball_base': 56, 'baseball': 57, 'baseball_bat': 58, 'baseball_cap': 59, 'baseball_glove': 60, 'basket': 61, 'basketball': 62, 'bass_horn': 63, 'bat_(animal)': 64, 'bath_mat': 65, 'bath_towel': 66, 'bathrobe': 67, 'bathtub': 68, 'batter_(food)': 69, 'battery': 70, 'beachball': 71, 'bead': 72, 'bean_curd': 73, 'beanbag': 74, 'beanie': 75, 'bear': 76, 'bed': 77, 'bedpan': 78, 'bedspread': 79, 'cow': 80, 'beef_(food)': 81, 'beeper': 82, 'beer_bottle': 83, 'beer_can': 84, 'beetle': 85, 'bell': 86, 'bell_pepper': 87, 'belt': 88, 'belt_buckle': 89, 'bench': 90, 'beret': 91, 'bib': 92, 'Bible': 93, 'bicycle': 94, 'visor': 95, 'billboard': 96, 'binder': 97, 'binoculars': 98, 'bird': 99, 'birdfeeder': 100, 'birdbath': 101, 'birdcage': 102, 'birdhouse': 103, 'birthday_cake': 104, 'birthday_card': 105, 'pirate_flag': 106, 'black_sheep': 107, 'blackberry': 108, 'blackboard': 109, 'blanket': 110, 'blazer': 111, 'blender': 112, 'blimp': 113, 'blinker': 114, 'blouse': 115, 'blueberry': 116, 'gameboard': 117, 'boat': 118, 'bob': 119, 'bobbin': 120, 'bobby_pin': 121, 'boiled_egg': 122, 'bolo_tie': 123, 'deadbolt': 124, 'bolt': 125, 'bonnet': 126, 'book': 127, 'bookcase': 128, 'booklet': 129, 'bookmark': 130, 'boom_microphone': 131, 'boot': 132, 'bottle': 133, 'bottle_opener': 134, 'bouquet': 135, 'bow_(weapon)': 136, 'bow_(decorative_ribbons)': 137, 'bow-tie': 138, 'bowl': 139, 'pipe_bowl': 140, 'bowler_hat': 141, 'bowling_ball': 142, 'box': 143, 'boxing_glove': 144, 'suspenders': 145, 'bracelet': 146, 'brass_plaque': 147, 'brassiere': 148, 'bread-bin': 149, 'bread': 150, 'breechcloth': 151, 'bridal_gown': 152, 'briefcase': 153, 'broccoli': 154, 'broach': 155, 'broom': 156, 'brownie': 157, 'brussels_sprouts': 158, 'bubble_gum': 159, 'bucket': 160, 'horse_buggy': 161, 'bull': 162, 'bulldog': 163, 'bulldozer': 164, 'bullet_train': 165, 'bulletin_board': 166, 'bulletproof_vest': 167, 'bullhorn': 168, 'bun': 169, 'bunk_bed': 170, 'buoy': 171, 'burrito': 172, 'bus_(vehicle)': 173, 'business_card': 174, 'butter': 175, 'butterfly': 176, 'button': 177, 'cab_(taxi)': 178, 'cabana': 179, 'cabin_car': 180, 'cabinet': 181, 'locker': 182, 'cake': 183, 'calculator': 184, 'calendar': 185, 'calf': 186, 'camcorder': 187, 'camel': 188, 'camera': 189, 'camera_lens': 190, 'camper_(vehicle)': 191, 'can': 192, 'can_opener': 193, 'candle': 194, 'candle_holder': 195, 'candy_bar': 196, 'candy_cane': 197, 'walking_cane': 198, 'canister': 199, 'canoe': 200, 'cantaloup': 201, 'canteen': 202, 'cap_(headwear)': 203, 'bottle_cap': 204, 'cape': 205, 'cappuccino': 206, 'car_(automobile)': 207, 'railcar_(part_of_a_train)': 208, 'elevator_car': 209, 'car_battery': 210, 'identity_card': 211, 'card': 212, 'cardigan': 213, 'cargo_ship': 214, 'carnation': 215, 'horse_carriage': 216, 'carrot': 217, 'tote_bag': 218, 'cart': 219, 'carton': 220, 'cash_register': 221, 'casserole': 222, 'cassette': 223, 'cast': 224, 'cat': 225, 'cauliflower': 226, 'cayenne_(spice)': 227, 'CD_player': 228, 'celery': 229, 'cellular_telephone': 230, 'chain_mail': 231, 'chair': 232, 'chaise_longue': 233, 'chalice': 234, 'chandelier': 235, 'chap': 236, 'checkbook': 237, 'checkerboard': 238, 'cherry': 239, 'chessboard': 240, 'chicken_(animal)': 241, 'chickpea': 242, 'chili_(vegetable)': 243, 'chime': 244, 'chinaware': 245, 'crisp_(potato_chip)': 246, 'poker_chip': 247, 'chocolate_bar': 248, 'chocolate_cake': 249, 'chocolate_milk': 250, 'chocolate_mousse': 251, 'choker': 252, 'chopping_board': 253, 'chopstick': 254, 'Christmas_tree': 255, 'slide': 256, 'cider': 257, 'cigar_box': 258, 'cigarette': 259, 'cigarette_case': 260, 'cistern': 261, 'clarinet': 262, 'clasp': 263, 'cleansing_agent': 264, 'cleat_(for_securing_rope)': 265, 'clementine': 266, 'clip': 267, 'clipboard': 268, 'clippers_(for_plants)': 269, 'cloak': 270, 'clock': 271, 'clock_tower': 272, 'clothes_hamper': 273, 'clothespin': 274, 'clutch_bag': 275, 'coaster': 276, 'coat': 277, 'coat_hanger': 278, 'coatrack': 279, 'cock': 280, 'cockroach': 281, 'cocoa_(beverage)': 282, 'coconut': 283, 'coffee_maker': 284, 'coffee_table': 285, 'coffeepot': 286, 'coil': 287, 'coin': 288, 'colander': 289, 'coleslaw': 290, 'coloring_material': 291, 'combination_lock': 292, 'pacifier': 293, 'comic_book': 294, 'compass': 295, 'computer_keyboard': 296, 'condiment': 297, 'cone': 298, 'control': 299, 'convertible_(automobile)': 300, 'sofa_bed': 301, 'cooker': 302, 'cookie': 303, 'cooking_utensil': 304, 'cooler_(for_food)': 305, 'cork_(bottle_plug)': 306, 'corkboard': 307, 'corkscrew': 308, 'edible_corn': 309, 'cornbread': 310, 'cornet': 311, 'cornice': 312, 'cornmeal': 313, 'corset': 314, 'costume': 315, 'cougar': 316, 'coverall': 317, 'cowbell': 318, 'cowboy_hat': 319, 'crab_(animal)': 320, 'crabmeat': 321, 'cracker': 322, 'crape': 323, 'crate': 324, 'crayon': 325, 'cream_pitcher': 326, 'crescent_roll': 327, 'crib': 328, 'crock_pot': 329, 'crossbar': 330, 'crouton': 331, 'crow': 332, 'crowbar': 333, 'crown': 334, 'crucifix': 335, 'cruise_ship': 336, 'police_cruiser': 337, 'crumb': 338, 'crutch': 339, 'cub_(animal)': 340, 'cube': 341, 'cucumber': 342, 'cufflink': 343, 'cup': 344, 'trophy_cup': 345, 'cupboard': 346, 'cupcake': 347, 'hair_curler': 348, 'curling_iron': 349, 'curtain': 350, 'cushion': 351, 'cylinder': 352, 'cymbal': 353, 'dagger': 354, 'dalmatian': 355, 'dartboard': 356, 'date_(fruit)': 357, 'deck_chair': 358, 'deer': 359, 'dental_floss': 360, 'desk': 361, 'detergent': 362, 'diaper': 363, 'diary': 364, 'die': 365, 'dinghy': 366, 'dining_table': 367, 'tux': 368, 'dish': 369, 'dish_antenna': 370, 'dishrag': 371, 'dishtowel': 372, 'dishwasher': 373, 'dishwasher_detergent': 374, 'dispenser': 375, 'diving_board': 376, 'Dixie_cup': 377, 'dog': 378, 'dog_collar': 379, 'doll': 380, 'dollar': 381, 'dollhouse': 382, 'dolphin': 383, 'domestic_ass': 384, 'doorknob': 385, 'doormat': 386, 'doughnut': 387, 'dove': 388, 'dragonfly': 389, 'drawer': 390, 'underdrawers': 391, 'dress': 392, 'dress_hat': 393, 'dress_suit': 394, 'dresser': 395, 'drill': 396, 'drone': 397, 'dropper': 398, 'drum_(musical_instrument)': 399, 'drumstick': 400, 'duck': 401, 'duckling': 402, 'duct_tape': 403, 'duffel_bag': 404, 'dumbbell': 405, 'dumpster': 406, 'dustpan': 407, 'eagle': 408, 'earphone': 409, 'earplug': 410, 'earring': 411, 'easel': 412, 'eclair': 413, 'eel': 414, 'egg': 415, 'egg_roll': 416, 'egg_yolk': 417, 'eggbeater': 418, 'eggplant': 419, 'electric_chair': 420, 'refrigerator': 421, 'elephant': 422, 'elk': 423, 'envelope': 424, 'eraser': 425, 'escargot': 426, 'eyepatch': 427, 'falcon': 428, 'fan': 429, 'faucet': 430, 'fedora': 431, 'ferret': 432, 'Ferris_wheel': 433, 'ferry': 434, 'fig_(fruit)': 435, 'fighter_jet': 436, 'figurine': 437, 'file_cabinet': 438, 'file_(tool)': 439, 'fire_alarm': 440, 'fire_engine': 441, 'fire_extinguisher': 442, 'fire_hose': 443, 'fireplace': 444, 'fireplug': 445, 'first-aid_kit': 446, 'fish': 447, 'fish_(food)': 448, 'fishbowl': 449, 'fishing_rod': 450, 'flag': 451, 'flagpole': 452, 'flamingo': 453, 'flannel': 454, 'flap': 455, 'flash': 456, 'flashlight': 457, 'fleece': 458, 'flip-flop_(sandal)': 459, 'flipper_(footwear)': 460, 'flower_arrangement': 461, 'flute_glass': 462, 'foal': 463, 'folding_chair': 464, 'food_processor': 465, 'football_(American)': 466, 'football_helmet': 467, 'footstool': 468, 'fork': 469, 'forklift': 470, 'freight_car': 471, 'French_toast': 472, 'freshener': 473, 'frisbee': 474, 'frog': 475, 'fruit_juice': 476, 'frying_pan': 477, 'fudge': 478, 'funnel': 479, 'futon': 480, 'gag': 481, 'garbage': 482, 'garbage_truck': 483, 'garden_hose': 484, 'gargle': 485, 'gargoyle': 486, 'garlic': 487, 'gasmask': 488, 'gazelle': 489, 'gelatin': 490, 'gemstone': 491, 'generator': 492, 'giant_panda': 493, 'gift_wrap': 494, 'ginger': 495, 'giraffe': 496, 'cincture': 497, 'glass_(drink_container)': 498, 'globe': 499, 'glove': 500, 'goat': 501, 'goggles': 502, 'goldfish': 503, 'golf_club': 504, 'golfcart': 505, 'gondola_(boat)': 506, 'goose': 507, 'gorilla': 508, 'gourd': 509, 'grape': 510, 'grater': 511, 'gravestone': 512, 'gravy_boat': 513, 'green_bean': 514, 'green_onion': 515, 'griddle': 516, 'grill': 517, 'grits': 518, 'grizzly': 519, 'grocery_bag': 520, 'guitar': 521, 'gull': 522, 'gun': 523, 'hairbrush': 524, 'hairnet': 525, 'hairpin': 526, 'halter_top': 527, 'ham': 528, 'hamburger': 529, 'hammer': 530, 'hammock': 531, 'hamper': 532, 'hamster': 533, 'hair_dryer': 534, 'hand_glass': 535, 'hand_towel': 536, 'handcart': 537, 'handcuff': 538, 'handkerchief': 539, 'handle': 540, 'handsaw': 541, 'hardback_book': 542, 'harmonium': 543, 'hat': 544, 'hatbox': 545, 'veil': 546, 'headband': 547, 'headboard': 548, 'headlight': 549, 'headscarf': 550, 'headset': 551, 'headstall_(for_horses)': 552, 'heart': 553, 'heater': 554, 'helicopter': 555, 'helmet': 556, 'heron': 557, 'highchair': 558, 'hinge': 559, 'hippopotamus': 560, 'hockey_stick': 561, 'hog': 562, 'home_plate_(baseball)': 563, 'honey': 564, 'fume_hood': 565, 'hook': 566, 'hookah': 567, 'hornet': 568, 'horse': 569, 'hose': 570, 'hot-air_balloon': 571, 'hotplate': 572, 'hot_sauce': 573, 'hourglass': 574, 'houseboat': 575, 'hummingbird': 576, 'hummus': 577, 'polar_bear': 578, 'icecream': 579, 'popsicle': 580, 'ice_maker': 581, 'ice_pack': 582, 'ice_skate': 583, 'igniter': 584, 'inhaler': 585, 'iPod': 586, 'iron_(for_clothing)': 587, 'ironing_board': 588, 'jacket': 589, 'jam': 590, 'jar': 591, 'jean': 592, 'jeep': 593, 'jelly_bean': 594, 'jersey': 595, 'jet_plane': 596, 'jewel': 597, 'jewelry': 598, 'joystick': 599, 'jumpsuit': 600, 'kayak': 601, 'keg': 602, 'kennel': 603, 'kettle': 604, 'key': 605, 'keycard': 606, 'kilt': 607, 'kimono': 608, 'kitchen_sink': 609, 'kitchen_table': 610, 'kite': 611, 'kitten': 612, 'kiwi_fruit': 613, 'knee_pad': 614, 'knife': 615, 'knitting_needle': 616, 'knob': 617, 'knocker_(on_a_door)': 618, 'koala': 619, 'lab_coat': 620, 'ladder': 621, 'ladle': 622, 'ladybug': 623, 'lamb_(animal)': 624, 'lamb-chop': 625, 'lamp': 626, 'lamppost': 627, 'lampshade': 628, 'lantern': 629, 'lanyard': 630, 'laptop_computer': 631, 'lasagna': 632, 'latch': 633, 'lawn_mower': 634, 'leather': 635, 'legging_(clothing)': 636, 'Lego': 637, 'legume': 638, 'lemon': 639, 'lemonade': 640, 'lettuce': 641, 'license_plate': 642, 'life_buoy': 643, 'life_jacket': 644, 'lightbulb': 645, 'lightning_rod': 646, 'lime': 647, 'limousine': 648, 'lion': 649, 'lip_balm': 650, 'liquor': 651, 'lizard': 652, 'log': 653, 'lollipop': 654, 'speaker_(stero_equipment)': 655, 'loveseat': 656, 'machine_gun': 657, 'magazine': 658, 'magnet': 659, 'mail_slot': 660, 'mailbox_(at_home)': 661, 'mallard': 662, 'mallet': 663, 'mammoth': 664, 'manatee': 665, 'mandarin_orange': 666, 'manger': 667, 'manhole': 668, 'map': 669, 'marker': 670, 'martini': 671, 'mascot': 672, 'mashed_potato': 673, 'masher': 674, 'mask': 675, 'mast': 676, 'mat_(gym_equipment)': 677, 'matchbox': 678, 'mattress': 679, 'measuring_cup': 680, 'measuring_stick': 681, 'meatball': 682, 'medicine': 683, 'melon': 684, 'microphone': 685, 'microscope': 686, 'microwave_oven': 687, 'milestone': 688, 'milk': 689, 'milk_can': 690, 'milkshake': 691, 'minivan': 692, 'mint_candy': 693, 'mirror': 694, 'mitten': 695, 'mixer_(kitchen_tool)': 696, 'money': 697, 'monitor_(computer_equipment)_computer_monitor': 698, 'monkey': 699, 'motor': 700, 'motor_scooter': 701, 'motor_vehicle': 702, 'motorcycle': 703, 'mound_(baseball)': 704, 'mouse_(computer_equipment)': 705, 'mousepad': 706, 'muffin': 707, 'mug': 708, 'mushroom': 709, 'music_stool': 710, 'musical_instrument': 711, 'nailfile': 712, 'napkin': 713, 'neckerchief': 714, 'necklace': 715, 'necktie': 716, 'needle': 717, 'nest': 718, 'newspaper': 719, 'newsstand': 720, 'nightshirt': 721, 'nosebag_(for_animals)': 722, 'noseband_(for_animals)': 723, 'notebook': 724, 'notepad': 725, 'nut': 726, 'nutcracker': 727, 'oar': 728, 'octopus_(food)': 729, 'octopus_(animal)': 730, 'oil_lamp': 731, 'olive_oil': 732, 'omelet': 733, 'onion': 734, 'orange_(fruit)': 735, 'orange_juice': 736, 'ostrich': 737, 'ottoman': 738, 'oven': 739, 'overalls_(clothing)': 740, 'owl': 741, 'packet': 742, 'inkpad': 743, 'pad': 744, 'paddle': 745, 'padlock': 746, 'paintbrush': 747, 'painting': 748, 'pajamas': 749, 'palette': 750, 'pan_(for_cooking)': 751, 'pan_(metal_container)': 752, 'pancake': 753, 'pantyhose': 754, 'papaya': 755, 'paper_plate': 756, 'paper_towel': 757, 'paperback_book': 758, 'paperweight': 759, 'parachute': 760, 'parakeet': 761, 'parasail_(sports)': 762, 'parasol': 763, 'parchment': 764, 'parka': 765, 'parking_meter': 766, 'parrot': 767, 'passenger_car_(part_of_a_train)': 768, 'passenger_ship': 769, 'passport': 770, 'pastry': 771, 'patty_(food)': 772, 'pea_(food)': 773, 'peach': 774, 'peanut_butter': 775, 'pear': 776, 'peeler_(tool_for_fruit_and_vegetables)': 777, 'wooden_leg': 778, 'pegboard': 779, 'pelican': 780, 'pen': 781, 'pencil': 782, 'pencil_box': 783, 'pencil_sharpener': 784, 'pendulum': 785, 'penguin': 786, 'pennant': 787, 'penny_(coin)': 788, 'pepper': 789, 'pepper_mill': 790, 'perfume': 791, 'persimmon': 792, 'person': 793, 'pet': 794, 'pew_(church_bench)': 795, 'phonebook': 796, 'phonograph_record': 797, 'piano': 798, 'pickle': 799, 'pickup_truck': 800, 'pie': 801, 'pigeon': 802, 'piggy_bank': 803, 'pillow': 804, 'pin_(non_jewelry)': 805, 'pineapple': 806, 'pinecone': 807, 'ping-pong_ball': 808, 'pinwheel': 809, 'tobacco_pipe': 810, 'pipe': 811, 'pistol': 812, 'pita_(bread)': 813, 'pitcher_(vessel_for_liquid)': 814, 'pitchfork': 815, 'pizza': 816, 'place_mat': 817, 'plate': 818, 'platter': 819, 'playpen': 820, 'pliers': 821, 'plow_(farm_equipment)': 822, 'plume': 823, 'pocket_watch': 824, 'pocketknife': 825, 'poker_(fire_stirring_tool)': 826, 'pole': 827, 'polo_shirt': 828, 'poncho': 829, 'pony': 830, 'pool_table': 831, 'pop_(soda)': 832, 'postbox_(public)': 833, 'postcard': 834, 'poster': 835, 'pot': 836, 'flowerpot': 837, 'potato': 838, 'potholder': 839, 'pottery': 840, 'pouch': 841, 'power_shovel': 842, 'prawn': 843, 'pretzel': 844, 'printer': 845, 'projectile_(weapon)': 846, 'projector': 847, 'propeller': 848, 'prune': 849, 'pudding': 850, 'puffer_(fish)': 851, 'puffin': 852, 'pug-dog': 853, 'pumpkin': 854, 'puncher': 855, 'puppet': 856, 'puppy': 857, 'quesadilla': 858, 'quiche': 859, 'quilt': 860, 'rabbit': 861, 'race_car': 862, 'racket': 863, 'radar': 864, 'radiator': 865, 'radio_receiver': 866, 'radish': 867, 'raft': 868, 'rag_doll': 869, 'raincoat': 870, 'ram_(animal)': 871, 'raspberry': 872, 'rat': 873, 'razorblade': 874, 'reamer_(juicer)': 875, 'rearview_mirror': 876, 'receipt': 877, 'recliner': 878, 'record_player': 879, 'reflector': 880, 'remote_control': 881, 'rhinoceros': 882, 'rib_(food)': 883, 'rifle': 884, 'ring': 885, 'river_boat': 886, 'road_map': 887, 'robe': 888, 'rocking_chair': 889, 'rodent': 890, 'roller_skate': 891, 'Rollerblade': 892, 'rolling_pin': 893, 'root_beer': 894, 'router_(computer_equipment)': 895, 'rubber_band': 896, 'runner_(carpet)': 897, 'plastic_bag': 898, 'saddle_(on_an_animal)': 899, 'saddle_blanket': 900, 'saddlebag': 901, 'safety_pin': 902, 'sail': 903, 'salad': 904, 'salad_plate': 905, 'salami': 906, 'salmon_(fish)': 907, 'salmon_(food)': 908, 'salsa': 909, 'saltshaker': 910, 'sandal_(type_of_shoe)': 911, 'sandwich': 912, 'satchel': 913, 'saucepan': 914, 'saucer': 915, 'sausage': 916, 'sawhorse': 917, 'saxophone': 918, 'scale_(measuring_instrument)': 919, 'scarecrow': 920, 'scarf': 921, 'school_bus': 922, 'scissors': 923, 'scoreboard': 924, 'scraper': 925, 'screwdriver': 926, 'scrubbing_brush': 927, 'sculpture': 928, 'seabird': 929, 'seahorse': 930, 'seaplane': 931, 'seashell': 932, 'sewing_machine': 933, 'shaker': 934, 'shampoo': 935, 'shark': 936, 'sharpener': 937, 'Sharpie': 938, 'shaver_(electric)': 939, 'shaving_cream': 940, 'shawl': 941, 'shears': 942, 'sheep': 943, 'shepherd_dog': 944, 'sherbert': 945, 'shield': 946, 'shirt': 947, 'shoe': 948, 'shopping_bag': 949, 'shopping_cart': 950, 'short_pants': 951, 'shot_glass': 952, 'shoulder_bag': 953, 'shovel': 954, 'shower_head': 955, 'shower_cap': 956, 'shower_curtain': 957, 'shredder_(for_paper)': 958, 'signboard': 959, 'silo': 960, 'sink': 961, 'skateboard': 962, 'skewer': 963, 'ski': 964, 'ski_boot': 965, 'ski_parka': 966, 'ski_pole': 967, 'skirt': 968, 'skullcap': 969, 'sled': 970, 'sleeping_bag': 971, 'sling_(bandage)': 972, 'slipper_(footwear)': 973, 'smoothie': 974, 'snake': 975, 'snowboard': 976, 'snowman': 977, 'snowmobile': 978, 'soap': 979, 'soccer_ball': 980, 'sock': 981, 'sofa': 982, 'softball': 983, 'solar_array': 984, 'sombrero': 985, 'soup': 986, 'soup_bowl': 987, 'soupspoon': 988, 'sour_cream': 989, 'soya_milk': 990, 'space_shuttle': 991, 'sparkler_(fireworks)': 992, 'spatula': 993, 'spear': 994, 'spectacles': 995, 'spice_rack': 996, 'spider': 997, 'crawfish': 998, 'sponge': 999, 'spoon': 1000, 'sportswear': 1001, 'spotlight': 1002, 'squid_(food)': 1003, 'squirrel': 1004, 'stagecoach': 1005, 'stapler_(stapling_machine)': 1006, 'starfish': 1007, 'statue_(sculpture)': 1008, 'steak_(food)': 1009, 'steak_knife': 1010, 'steering_wheel': 1011, 'stepladder': 1012, 'step_stool': 1013, 'stereo_(sound_system)': 1014, 'stew': 1015, 'stirrer': 1016, 'stirrup': 1017, 'stool': 1018, 'stop_sign': 1019, 'brake_light': 1020, 'stove': 1021, 'strainer': 1022, 'strap': 1023, 'straw_(for_drinking)': 1024, 'strawberry': 1025, 'street_sign': 1026, 'streetlight': 1027, 'string_cheese': 1028, 'stylus': 1029, 'subwoofer': 1030, 'sugar_bowl': 1031, 'sugarcane_(plant)': 1032, 'suit_(clothing)': 1033, 'sunflower': 1034, 'sunglasses': 1035, 'sunhat': 1036, 'surfboard': 1037, 'sushi': 1038, 'mop': 1039, 'sweat_pants': 1040, 'sweatband': 1041, 'sweater': 1042, 'sweatshirt': 1043, 'sweet_potato': 1044, 'swimsuit': 1045, 'sword': 1046, 'syringe': 1047, 'Tabasco_sauce': 1048, 'table-tennis_table': 1049, 'table': 1050, 'table_lamp': 1051, 'tablecloth': 1052, 'tachometer': 1053, 'taco': 1054, 'tag': 1055, 'taillight': 1056, 'tambourine': 1057, 'army_tank': 1058, 'tank_(storage_vessel)': 1059, 'tank_top_(clothing)': 1060, 'tape_(sticky_cloth_or_paper)': 1061, 'tape_measure': 1062, 'tapestry': 1063, 'tarp': 1064, 'tartan': 1065, 'tassel': 1066, 'tea_bag': 1067, 'teacup': 1068, 'teakettle': 1069, 'teapot': 1070, 'teddy_bear': 1071, 'telephone': 1072, 'telephone_booth': 1073, 'telephone_pole': 1074, 'telephoto_lens': 1075, 'television_camera': 1076, 'television_set': 1077, 'tennis_ball': 1078, 'tennis_racket': 1079, 'tequila': 1080, 'thermometer': 1081, 'thermos_bottle': 1082, 'thermostat': 1083, 'thimble': 1084, 'thread': 1085, 'thumbtack': 1086, 'tiara': 1087, 'tiger': 1088, 'tights_(clothing)': 1089, 'timer': 1090, 'tinfoil': 1091, 'tinsel': 1092, 'tissue_paper': 1093, 'toast_(food)': 1094, 'toaster': 1095, 'toaster_oven': 1096, 'toilet': 1097, 'toilet_tissue': 1098, 'tomato': 1099, 'tongs': 1100, 'toolbox': 1101, 'toothbrush': 1102, 'toothpaste': 1103, 'toothpick': 1104, 'cover': 1105, 'tortilla': 1106, 'tow_truck': 1107, 'towel': 1108, 'towel_rack': 1109, 'toy': 1110, 'tractor_(farm_equipment)': 1111, 'traffic_light': 1112, 'dirt_bike': 1113, 'trailer_truck': 1114, 'train_(railroad_vehicle)': 1115, 'trampoline': 1116, 'tray': 1117, 'trench_coat': 1118, 'triangle_(musical_instrument)': 1119, 'tricycle': 1120, 'tripod': 1121, 'trousers': 1122, 'truck': 1123, 'truffle_(chocolate)': 1124, 'trunk': 1125, 'vat': 1126, 'turban': 1127, 'turkey_(food)': 1128, 'turnip': 1129, 'turtle': 1130, 'turtleneck_(clothing)': 1131, 'typewriter': 1132, 'umbrella': 1133, 'underwear': 1134, 'unicycle': 1135, 'urinal': 1136, 'urn': 1137, 'vacuum_cleaner': 1138, 'vase': 1139, 'vending_machine': 1140, 'vent': 1141, 'vest': 1142, 'videotape': 1143, 'vinegar': 1144, 'violin': 1145, 'vodka': 1146, 'volleyball': 1147, 'vulture': 1148, 'waffle': 1149, 'waffle_iron': 1150, 'wagon': 1151, 'wagon_wheel': 1152, 'walking_stick': 1153, 'wall_clock': 1154, 'wall_socket': 1155, 'wallet': 1156, 'walrus': 1157, 'wardrobe': 1158, 'washbasin': 1159, 'automatic_washer': 1160, 'watch': 1161, 'water_bottle': 1162, 'water_cooler': 1163, 'water_faucet': 1164, 'water_heater': 1165, 'water_jug': 1166, 'water_gun': 1167, 'water_scooter': 1168, 'water_ski': 1169, 'water_tower': 1170, 'watering_can': 1171, 'watermelon': 1172, 'weathervane': 1173, 'webcam': 1174, 'wedding_cake': 1175, 'wedding_ring': 1176, 'wet_suit': 1177, 'wheel': 1178, 'wheelchair': 1179, 'whipped_cream': 1180, 'whistle': 1181, 'wig': 1182, 'wind_chime': 1183, 'windmill': 1184, 'window_box_(for_plants)': 1185, 'windshield_wiper': 1186, 'windsock': 1187, 'wine_bottle': 1188, 'wine_bucket': 1189, 'wineglass': 1190, 'blinder_(for_horses)': 1191, 'wok': 1192, 'wolf': 1193, 'wooden_spoon': 1194, 'wreath': 1195, 'wrench': 1196, 'wristband': 1197, 'wristlet': 1198, 'yacht': 1199, 'yogurt': 1200, 'yoke_(animal_equipment)': 1201, 'zebra': 1202, 'zucchini': 1203}

#{'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
class2type = {type2class[t]:t for t in type2class}


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2


class SUNObject3d(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.xmin = data[1] 
        self.ymin = data[2]
        self.xmax = data[1]+data[3]
        self.ymax = data[2]+data[4]
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        self.centroid = np.array([data[5],data[6],data[7]])
        self.unused_dimension = np.array([data[8],data[9],data[10]])
        self.w = data[8]
        self.l = data[9]
        self.h = data[10]
        self.orientation = np.zeros((3,))
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.heading_angle = -1 * np.arctan2(self.orientation[1], self.orientation[0])

class SUNRGBD_Calibration(object):
    ''' Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''

    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3,3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3,3), order='F')
        self.f_u = self.K[0,0]
        self.f_v = self.K[1,1]
        self.c_u = self.K[0,2]
        self.c_v = self.K[1,2]
   
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:,0:3])) # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K)) # (n,3)
        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]
        return uv[:,0:2], pc2[:,2]

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)

 
 
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """ 
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_sunrgbd_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [SUNObject3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_depth_points(depth_filename):
    depth = np.loadtxt(depth_filename)
    return depth

def load_depth_points_mat(depth_filename):
    depth = sio.loadmat(depth_filename)['instance']
    return depth

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
 
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)


def compute_box_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = rotz(-1*obj.heading_angle)
    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = obj.l # along heading arrow
    w = obj.w # perpendicular to heading arrow
    h = obj.h

    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d,_ = calib.project_upright_depth_to_image(np.transpose(corners_3d))
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_orientation_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in image coord.
            orientation_3d: (2,3) array in depth coord.
    '''
    
    # orientation in object coordinate system
    ori = obj.orientation
    orientation_3d = np.array([[0, ori[0]],[0, ori[1]],[0,0]])
    center = obj.centroid
    orientation_3d[0,:] = orientation_3d[0,:] + center[0]
    orientation_3d[1,:] = orientation_3d[1,:] + center[1]
    orientation_3d[2,:] = orientation_3d[2,:] + center[2]
    
    # project orientation into the image plane
    orientation_2d,_ = calib.project_upright_depth_to_image(np.transpose(orientation_3d))
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
    return image


import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
