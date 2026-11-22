v {xschem version=3.4.4 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
C {devices/code_shown.sym} 150 100 0 0 {name=MODELS only_toplevel=false value="* SKY130A typical-corner thick-oxide models
.option scale=1u
.include /home/fatdog/.ciel/sky130A/libs.tech/ngspice/parameters/lod.spice
.param mc_mm_switch=0 mc_pr_switch=0 hv_dlc_rotweak=0
.param sky130_fd_pr__nfet_03v3_nvt__dlc_rotweak=hv_dlc_rotweak
.param sky130_fd_pr__pfet_g5v0d10v5__dlc_rotweak=hv_dlc_rotweak
.include /home/fatdog/.ciel/sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__nfet_03v3_nvt__tt.corner.spice
.include /home/fatdog/.ciel/sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__nfet_03v3_nvt__mismatch.corner.spice
.include /home/fatdog/.ciel/sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__pfet_g5v0d10v5__tt.corner.spice
.include /home/fatdog/.ciel/sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__pfet_g5v0d10v5__mismatch.corner.spice
.control
op
print v(psrc) v(nsrc) v(pmid1) v(pmid2) v(upper1) v(lower1) v(lower2) i(VDD) i(VOUT)
print @m.xm3.msky130_fd_pr__nfet_03v3_nvt[id] @m.xm5.msky130_fd_pr__pfet_g5v0d10v5[id]
quit
.endc"}

T {3.3 V complementary folded-cascode OTA -- starting bias point, not optimized} 500 20 0 0 0.35 0.35 {}
T {3.3 V NMOS: nfet_03v3_nvt; PMOS: 5 V-rated pfet_g5v0d10v5 (no 3.3 V PMOS in installed PDK)} 500 60 0 0 0.25 0.25 {}
T {VDD=3.3 V; VCM=VOUT=1.65 V; VB1=1.50 V, VB2=0.95 V, VB3=0.10 V; tail currents=100 uA} 500 90 0 0 0.25 0.25 {}

C {sky130_fd_pr/pfet_g5v0d10v5.sym} 450 400 0 0 {name=M1 model=pfet_g5v0d10v5 W=50 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/pfet_g5v0d10v5.sym} 700 400 0 0 {name=M2 model=pfet_g5v0d10v5 W=50 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/nfet_03v3_nvt.sym} 450 650 0 0 {name=M3 model=nfet_03v3_nvt W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/nfet_03v3_nvt.sym} 700 650 0 0 {name=M4 model=nfet_03v3_nvt W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/pfet_g5v0d10v5.sym} 950 150 0 0 {name=M5 model=pfet_g5v0d10v5 W=12.5 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/pfet_g5v0d10v5.sym} 1200 150 0 0 {name=M6 model=pfet_g5v0d10v5 W=12.5 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/pfet_g5v0d10v5.sym} 950 350 0 0 {name=M7 model=pfet_g5v0d10v5 W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/pfet_g5v0d10v5.sym} 1200 350 0 0 {name=M8 model=pfet_g5v0d10v5 W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/nfet_03v3_nvt.sym} 450 900 0 0 {name=M9 model=nfet_03v3_nvt W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/nfet_03v3_nvt.sym} 700 900 0 0 {name=M10 model=nfet_03v3_nvt W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/nfet_03v3_nvt.sym} 450 1100 0 0 {name=M11 model=nfet_03v3_nvt W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}
C {sky130_fd_pr/nfet_03v3_nvt.sym} 700 1100 0 0 {name=M12 model=nfet_03v3_nvt W=10 L=0.5 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0}

* Each pin is given a global net label so the topology is easy to edit in Xschem.
C {devices/lab_pin.sym} 470 370 0 0 {name=l_m1_s lab=psrc}
C {devices/lab_pin.sym} 430 400 0 0 {name=l_m1_g lab=inp}
C {devices/lab_pin.sym} 470 430 0 0 {name=l_m1_d lab=lower1}
C {devices/lab_pin.sym} 470 400 0 0 {name=l_m1_b lab=vdd}
C {devices/lab_pin.sym} 720 370 0 0 {name=l_m2_s lab=psrc}
C {devices/lab_pin.sym} 680 400 0 0 {name=l_m2_g lab=inn}
C {devices/lab_pin.sym} 720 430 0 0 {name=l_m2_d lab=lower2}
C {devices/lab_pin.sym} 720 400 0 0 {name=l_m2_b lab=vdd}
C {devices/lab_pin.sym} 470 620 0 0 {name=l_m3_d lab=pmid1}
C {devices/lab_pin.sym} 430 650 0 0 {name=l_m3_g lab=inp}
C {devices/lab_pin.sym} 470 680 0 0 {name=l_m3_s lab=nsrc}
C {devices/lab_pin.sym} 470 650 0 0 {name=l_m3_b lab=0}
C {devices/lab_pin.sym} 720 620 0 0 {name=l_m4_d lab=out}
C {devices/lab_pin.sym} 680 650 0 0 {name=l_m4_g lab=inn}
C {devices/lab_pin.sym} 720 680 0 0 {name=l_m4_s lab=nsrc}
C {devices/lab_pin.sym} 720 650 0 0 {name=l_m4_b lab=0}
C {devices/lab_pin.sym} 970 120 0 0 {name=l_m5_s lab=vdd}
C {devices/lab_pin.sym} 930 150 0 0 {name=l_m5_g lab=upper1}
C {devices/lab_pin.sym} 970 180 0 0 {name=l_m5_d lab=pmid1}
C {devices/lab_pin.sym} 970 150 0 0 {name=l_m5_b lab=vdd}
C {devices/lab_pin.sym} 1220 120 0 0 {name=l_m6_s lab=vdd}
C {devices/lab_pin.sym} 1180 150 0 0 {name=l_m6_g lab=upper1}
C {devices/lab_pin.sym} 1220 180 0 0 {name=l_m6_d lab=pmid2}
C {devices/lab_pin.sym} 1220 150 0 0 {name=l_m6_b lab=vdd}
C {devices/lab_pin.sym} 970 320 0 0 {name=l_m7_s lab=pmid1}
C {devices/lab_pin.sym} 930 350 0 0 {name=l_m7_g lab=vb1}
C {devices/lab_pin.sym} 970 380 0 0 {name=l_m7_d lab=upper1}
C {devices/lab_pin.sym} 970 350 0 0 {name=l_m7_b lab=pmid1}
C {devices/lab_pin.sym} 1220 320 0 0 {name=l_m8_s lab=pmid2}
C {devices/lab_pin.sym} 1180 350 0 0 {name=l_m8_g lab=vb1}
C {devices/lab_pin.sym} 1220 380 0 0 {name=l_m8_d lab=out}
C {devices/lab_pin.sym} 1220 350 0 0 {name=l_m8_b lab=pmid2}
C {devices/lab_pin.sym} 470 870 0 0 {name=l_m9_d lab=upper1}
C {devices/lab_pin.sym} 430 900 0 0 {name=l_m9_g lab=vb2}
C {devices/lab_pin.sym} 470 930 0 0 {name=l_m9_s lab=lower1}
C {devices/lab_pin.sym} 470 900 0 0 {name=l_m9_b lab=0}
C {devices/lab_pin.sym} 720 870 0 0 {name=l_m10_d lab=out}
C {devices/lab_pin.sym} 680 900 0 0 {name=l_m10_g lab=vb2}
C {devices/lab_pin.sym} 720 930 0 0 {name=l_m10_s lab=lower2}
C {devices/lab_pin.sym} 720 900 0 0 {name=l_m10_b lab=0}
C {devices/lab_pin.sym} 470 1070 0 0 {name=l_m11_d lab=lower1}
C {devices/lab_pin.sym} 430 1100 0 0 {name=l_m11_g lab=vb3}
C {devices/lab_pin.sym} 470 1130 0 0 {name=l_m11_s lab=0}
C {devices/lab_pin.sym} 470 1100 0 0 {name=l_m11_b lab=0}
C {devices/lab_pin.sym} 720 1070 0 0 {name=l_m12_d lab=lower2}
C {devices/lab_pin.sym} 680 1100 0 0 {name=l_m12_g lab=vb3}
C {devices/lab_pin.sym} 720 1130 0 0 {name=l_m12_s lab=0}
C {devices/lab_pin.sym} 720 1100 0 0 {name=l_m12_b lab=0}

C {devices/vsource.sym} 1500 250 0 0 {name=VDD value=3.3}
C {devices/isource.sym} 1500 400 0 0 {name=IP value=100u}
C {devices/isource.sym} 1650 400 0 0 {name=IN value=100u}
C {devices/vsource.sym} 1500 600 0 0 {name=VOUT value=1.65}
C {devices/vsource.sym} 1500 750 0 0 {name=VB1 value=1.5}
C {devices/vsource.sym} 1500 900 0 0 {name=VB2 value=0.95}
C {devices/vsource.sym} 1500 1050 0 0 {name=VB3 value=0.1}
C {devices/vsource.sym} 1800 750 0 0 {name=VINP value="dc 1.65 ac 1"}
C {devices/vsource.sym} 1800 900 0 0 {name=VINN value=1.65}

C {devices/lab_pin.sym} 1500 220 0 0 {name=l_vdd_src lab=vdd}
C {devices/lab_pin.sym} 1500 280 0 0 {name=l_vdd_gnd lab=0}
C {devices/lab_pin.sym} 1500 370 0 0 {name=l_ip_hi lab=vdd}
C {devices/lab_pin.sym} 1500 430 0 0 {name=l_ip_lo lab=psrc}
C {devices/lab_pin.sym} 1650 370 0 0 {name=l_in_hi lab=nsrc}
C {devices/lab_pin.sym} 1650 430 0 0 {name=l_in_lo lab=0}
C {devices/lab_pin.sym} 1500 570 0 0 {name=l_vo_hi lab=out}
C {devices/lab_pin.sym} 1500 630 0 0 {name=l_vo_lo lab=0}
C {devices/lab_pin.sym} 1500 720 0 0 {name=l_vb1_hi lab=vb1}
C {devices/lab_pin.sym} 1500 780 0 0 {name=l_vb1_lo lab=0}
C {devices/lab_pin.sym} 1500 870 0 0 {name=l_vb2_hi lab=vb2}
C {devices/lab_pin.sym} 1500 930 0 0 {name=l_vb2_lo lab=0}
C {devices/lab_pin.sym} 1500 1020 0 0 {name=l_vb3_hi lab=vb3}
C {devices/lab_pin.sym} 1500 1080 0 0 {name=l_vb3_lo lab=0}
C {devices/lab_pin.sym} 1800 720 0 0 {name=l_vinp_hi lab=inp}
C {devices/lab_pin.sym} 1800 780 0 0 {name=l_vinp_lo lab=0}
C {devices/lab_pin.sym} 1800 870 0 0 {name=l_vinn_hi lab=inn}
C {devices/lab_pin.sym} 1800 930 0 0 {name=l_vinn_lo lab=0}
