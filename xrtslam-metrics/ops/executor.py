#!/usr/bin/env python

"""
Python script to execute multiple shell commands in parallel up to some given
number of concurrent jobs.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor

MAX_CONCURRENT_PROCESSES = 16

commands = [
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC01  -o MIC01_camcalib1 $msdmi/MIC_calibration/MIC01_camcalib1/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC02  -o MIC02_camcalib2 $msdmi/MIC_calibration/MIC02_camcalib2/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC03  -o MIC03_camcalib3 $msdmi/MIC_calibration/MIC03_camcalib3/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC04  -o MIC04_imucalib1 $msdmi/MIC_calibration/MIC04_imucalib1/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC05  -o MIC05_imucalib2 $msdmi/MIC_calibration/MIC05_imucalib2/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC06  -o MIC06_imucalib3 $msdmi/MIC_calibration/MIC06_imucalib3/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC07  -o MIC07_camcalib4 $msdmi/MIC_calibration/MIC07_camcalib4/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC08  -o MIC08_camcalib5 $msdmi/MIC_calibration/MIC08_camcalib5/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC09  -o MIC09_imucalib4 $msdmi/MIC_calibration/MIC09_imucalib4/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC10  -o MIC10_imucalib5 $msdmi/MIC_calibration/MIC10_imucalib5/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC11  -o MIC11_camcalib6 $msdmi/MIC_calibration/MIC11_camcalib6/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC12  -o MIC12_imucalib6 $msdmi/MIC_calibration/MIC12_imucalib6/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC13  -o MIC13_camcalib7 $msdmi/MIC_calibration/MIC13_camcalib7/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC14  -o MIC14_camcalib8 $msdmi/MIC_calibration/MIC14_camcalib8/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC15  -o MIC15_imucalib7 $msdmi/MIC_calibration/MIC15_imucalib7/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIC16  -o MIC16_imucalib8 $msdmi/MIC_calibration/MIC16_imucalib8/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO01  -o MIO01_hand_puncher_1 $msdmi/MIO_others/MIO01_hand_puncher_1/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO02  -o MIO02_hand_puncher_2 $msdmi/MIO_others/MIO02_hand_puncher_2/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO03  -o MIO03_hand_shooter_easy $msdmi/MIO_others/MIO03_hand_shooter_easy/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO04  -o MIO04_hand_shooter_hard $msdmi/MIO_others/MIO04_hand_shooter_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO05  -o MIO05_inspect_easy $msdmi/MIO_others/MIO05_inspect_easy/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO06  -o MIO06_inspect_hard $msdmi/MIO_others/MIO06_inspect_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO07  -o MIO07_mapping_easy $msdmi/MIO_others/MIO07_mapping_easy/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO08  -o MIO08_mapping_hard $msdmi/MIO_others/MIO08_mapping_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO09  -o MIO09_short_1_updown $msdmi/MIO_others/MIO09_short_1_updown/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO10  -o MIO10_short_2_panorama $msdmi/MIO_others/MIO10_short_2_panorama/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO11  -o MIO11_short_3_backandforth $msdmi/MIO_others/MIO11_short_3_backandforth/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO12  -o MIO12_moving_screens $msdmi/MIO_others/MIO12_moving_screens/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO13  -o MIO13_moving_person $msdmi/MIO_others/MIO13_moving_person/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO14  -o MIO14_moving_props $msdmi/MIO_others/MIO14_moving_props/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO15  -o MIO15_moving_person_props $msdmi/MIO_others/MIO15_moving_person_props/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIO16  -o MIO16_moving_screens_person_props $msdmi/MIO_others/MIO16_moving_screens_person_props/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB01 -o MIPB01_beatsaber_100bills_360_normal $msdmi/MIP_playing/MIPB_beat_saber/MIPB01_beatsaber_100bills_360_normal/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB02 -o MIPB02_beatsaber_crabrave_360_hard $msdmi/MIP_playing/MIPB_beat_saber/MIPB02_beatsaber_crabrave_360_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB03 -o MIPB03_beatsaber_countryrounds_360_expert $msdmi/MIP_playing/MIPB_beat_saber/MIPB03_beatsaber_countryrounds_360_expert/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB04 -o MIPB04_beatsaber_fitbeat_hard $msdmi/MIP_playing/MIPB_beat_saber/MIPB04_beatsaber_fitbeat_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB05 -o MIPB05_beatsaber_fitbeat_360_expert $msdmi/MIP_playing/MIPB_beat_saber/MIPB05_beatsaber_fitbeat_360_expert/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB06 -o MIPB06_beatsaber_fitbeat_expertplus_1 $msdmi/MIP_playing/MIPB_beat_saber/MIPB06_beatsaber_fitbeat_expertplus_1/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB07 -o MIPB07_beatsaber_fitbeat_expertplus_2 $msdmi/MIP_playing/MIPB_beat_saber/MIPB07_beatsaber_fitbeat_expertplus_2/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPB08 -o MIPB08_beatsaber_long_session_1 $msdmi/MIP_playing/MIPB_beat_saber/MIPB08_beatsaber_long_session_1/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPP01 -o MIPP01_pistolwhip_blackmagic_hard $msdmi/MIP_playing/MIPP_pistol_whip/MIPP01_pistolwhip_blackmagic_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPP02 -o MIPP02_pistolwhip_lilith_hard $msdmi/MIP_playing/MIPP_pistol_whip/MIPP02_pistolwhip_lilith_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPP03 -o MIPP03_pistolwhip_requiem_hard $msdmi/MIP_playing/MIPP_pistol_whip/MIPP03_pistolwhip_requiem_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPP04 -o MIPP04_pistolwhip_revelations_hard $msdmi/MIP_playing/MIPP_pistol_whip/MIPP04_pistolwhip_revelations_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPP05 -o MIPP05_pistolwhip_thefall_hard_2pistols $msdmi/MIP_playing/MIPP_pistol_whip/MIPP05_pistolwhip_thefall_hard_2pistols/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPP06 -o MIPP06_pistolwhip_thegrave_hard $msdmi/MIP_playing/MIPP_pistol_whip/MIPP06_pistolwhip_thegrave_hard/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPT01 -o MIPT01_thrillofthefight_setup $msdmi/MIP_playing/MIPT_thrill_of_the_fight/MIPT01_thrillofthefight_setup/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPT02 -o MIPT02_thrillofthefight_fight_1 $msdmi/MIP_playing/MIPT_thrill_of_the_fight/MIPT02_thrillofthefight_fight_1/",
    "$xrtmet/ops/euroc_ops.py preview_video -s 5 -t TMP_MIPT03 -o MIPT03_thrillofthefight_fight_2 $msdmi/MIP_playing/MIPT_thrill_of_the_fight/MIPT03_thrillofthefight_fight_2/",
]


def execute_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return f"Command '{command}' failed with error code {e.returncode}."
    except Exception as e:
        return f"An error occurred while executing '{command}': {str(e)}"


with ThreadPoolExecutor(MAX_CONCURRENT_PROCESSES) as executor:
    results = executor.map(execute_command, commands)

for result in results:
    print(result)
