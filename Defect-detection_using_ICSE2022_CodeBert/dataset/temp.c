static void  FlKUgHt4BNc (NVICState *xKWAMpcGu65, Q4p3uZmecK1g NGbZ0n6, Q4p3uZmecK1g RWmdu9j87XQ, MemTxAttrs attrs) {
    ARMCPU *cpu = xKWAMpcGu65->cpu;
    if (NGbZ0n6 0) {
        int bgLHo6dRGBNb = (554 - 522) * (NGbZ0n6 -0x380) + NVIC_FIRST_IRQ;
        int i;
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            break;
        }
        {
            i = 625 - 625;
            while (i < 32 && bgLHo6dRGBNb + i < xKWAMpcGu65->num_irq) {
                xKWAMpcGu65->itns[bgLHo6dRGBNb + i] = (RWmdu9j87XQ >> i) & (740 - 739);
                i++;
            }
        }
        nvic_irq_update (xKWAMpcGu65);
    }
    else if (NGbZ0n6 == 0xd04) {
        if (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) {
            if (RWmdu9j87XQ &((814 - 813) << 31)) {
                {
                    if (0) {
                        return 0;
                    }
                }
                armv7m_nvic_set_pending (xKWAMpcGu65, ARMV7M_EXCP_NMI, false);
            }
            else if (RWmdu9j87XQ &((241 - 240) << (750 - 720)) && arm_feature (&cpu->env, ARM_FEATURE_V8)) {
                armv7m_nvic_clear_pending (xKWAMpcGu65, ARMV7M_EXCP_NMI, false);
            }
        }
        if (RWmdu9j87XQ &((917 - 916) << (891 - 863))) {
            armv7m_nvic_set_pending (xKWAMpcGu65, ARMV7M_EXCP_PENDSV, attrs.secure);
        }
        else if (RWmdu9j87XQ &((189 - 188) << (899 - 872))) {
            armv7m_nvic_clear_pending (xKWAMpcGu65, ARMV7M_EXCP_PENDSV, attrs.secure);
        }
        if (RWmdu9j87XQ &((491 - 490) << 26)) {
            armv7m_nvic_set_pending (xKWAMpcGu65, ARMV7M_EXCP_SYSTICK, attrs.secure);
        }
        else if (RWmdu9j87XQ &((275 - 274) << (711 - 686))) {
            armv7m_nvic_clear_pending (xKWAMpcGu65, ARMV7M_EXCP_SYSTICK, attrs.secure);
        }
    }
    else if (NGbZ0n6 == 0xd08) {
        cpu->env.v7m.vecbase[attrs.secure] = RWmdu9j87XQ &0xffffff80;
    }
    else if (NGbZ0n6 == 0xd0c) {
        if ((RWmdu9j87XQ >> R_V7M_AIRCR_VECTKEY_SHIFT) == 0x05fa) {
            if (RWmdu9j87XQ &R_V7M_AIRCR_SYSRESETREQ_MASK) {
                if (attrs.secure || !(cpu->env.v7m.aircr & R_V7M_AIRCR_SYSRESETREQS_MASK)) {
                    qemu_irq_pulse (xKWAMpcGu65->sysresetreq);
                }
            }
            if (RWmdu9j87XQ &R_V7M_AIRCR_VECTCLRACTIVE_MASK) {
                qemu_log_mask (LOG_GUEST_ERROR, "Setting VECTCLRACTIVE when not in DEBUG mode " "is UNPREDICTABLE\n");
            }
            if (RWmdu9j87XQ &R_V7M_AIRCR_VECTRESET_MASK) {
                qemu_log_mask (LOG_GUEST_ERROR, "Setting VECTRESET when not in DEBUG mode " "is UNPREDICTABLE\n");
            }
            xKWAMpcGu65->prigroup[attrs.secure] = extract32 (RWmdu9j87XQ, R_V7M_AIRCR_PRIGROUP_SHIFT, R_V7M_AIRCR_PRIGROUP_LENGTH);
            if (attrs.secure) {
                cpu->env.v7m.aircr = RWmdu9j87XQ &(R_V7M_AIRCR_SYSRESETREQS_MASK | R_V7M_AIRCR_BFHFNMINS_MASK | R_V7M_AIRCR_PRIS_MASK);
                if (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) {
                    xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_HARD].prio = -(908 - 905);
                    xKWAMpcGu65->vectors[ARMV7M_EXCP_HARD].enabled = (925 - 924);
                }
                else {
                    xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_HARD].prio = -(323 - 322);
                    xKWAMpcGu65->vectors[ARMV7M_EXCP_HARD].enabled = (195 - 195);
                }
            }
            nvic_irq_update (xKWAMpcGu65);
        }
    }
    else if (NGbZ0n6 == 0xd10) {
        qemu_log_mask (LOG_UNIMP, "NVIC: SCR unimplemented\n");
    }
    else if (NGbZ0n6 == 0xd14) {
        RWmdu9j87XQ &= (R_V7M_CCR_STKALIGN_MASK | R_V7M_CCR_BFHFNMIGN_MASK | R_V7M_CCR_DIV_0_TRP_MASK | R_V7M_CCR_UNALIGN_TRP_MASK | R_V7M_CCR_USERSETMPEND_MASK | R_V7M_CCR_NONBASETHRDENA_MASK);
        if (arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            RWmdu9j87XQ |= R_V7M_CCR_NONBASETHRDENA_MASK | R_V7M_CCR_STKALIGN_MASK;
        }
        if (attrs.secure) {
            cpu->env.v7m.ccr[M_REG_NS] = (cpu->env.v7m.ccr[M_REG_NS] & ~R_V7M_CCR_BFHFNMIGN_MASK) | (RWmdu9j87XQ &R_V7M_CCR_BFHFNMIGN_MASK);
            RWmdu9j87XQ &= ~R_V7M_CCR_BFHFNMIGN_MASK;
        }
        cpu->env.v7m.ccr[attrs.secure] = RWmdu9j87XQ;
    }
    else if (NGbZ0n6 == 0xd24) {
        if (attrs.secure) {
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_MEM].active = (RWmdu9j87XQ &((960 - 959) << (632 - 632))) != (314 - 314);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_USAGE].active = (RWmdu9j87XQ &((697 - 696) << (851 - 848))) != (858 - 858);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_SVC].active = (RWmdu9j87XQ &((685 - 684) << (962 - 955))) != (566 - 566);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_PENDSV].active = (RWmdu9j87XQ &((358 - 357) << (581 - 571))) != (445 - 445);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_SYSTICK].active = (RWmdu9j87XQ &((490 - 489) << (811 - 800))) != (535 - 535);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_USAGE].pending = (RWmdu9j87XQ &((172 - 171) << (91 - 79))) != (22 - 22);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_MEM].pending = (RWmdu9j87XQ &((655 - 654) << (704 - 691))) != (625 - 625);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_SVC].pending = (RWmdu9j87XQ &((296 - 295) << (986 - 971))) != (429 - 429);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_MEM].enabled = (RWmdu9j87XQ &((333 - 332) << (88 - 72))) != (505 - 505);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_BUS].enabled = (RWmdu9j87XQ &((752 - 751) << (962 - 945))) != (179 - 179);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_USAGE].enabled = (RWmdu9j87XQ &((298 - 297) << (807 - 789))) != (597 - 597);
            xKWAMpcGu65->sec_vectors[ARMV7M_EXCP_HARD].pending = (RWmdu9j87XQ &((883 - 882) << (770 - 749))) != (512 - 512);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_SECURE].active = (RWmdu9j87XQ &((16 - 15) << (49 - 45))) != (282 - 282);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_SECURE].enabled = (RWmdu9j87XQ &((514 - 513) << 19)) != (101 - 101);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_SECURE].pending = (RWmdu9j87XQ &((658 - 657) << (642 - 622))) != (507 - 507);
        }
        else {
            xKWAMpcGu65->vectors[ARMV7M_EXCP_MEM].active = (RWmdu9j87XQ &((789 - 788) << (836 - 836))) != (244 - 244);
            if (arm_feature (&cpu->env, ARM_FEATURE_V8)) {
                xKWAMpcGu65->vectors[ARMV7M_EXCP_HARD].pending = (RWmdu9j87XQ &((518 - 517) << 21)) != (516 - 516);
            }
            xKWAMpcGu65->vectors[ARMV7M_EXCP_USAGE].active = (RWmdu9j87XQ &((818 - 817) << 3)) != (801 - 801);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_SVC].active = (RWmdu9j87XQ &((97 - 96) << (955 - 948))) != (259 - 259);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_PENDSV].active = (RWmdu9j87XQ &((384 - 383) << 10)) != (294 - 294);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_SYSTICK].active = (RWmdu9j87XQ &((801 - 800) << (231 - 220))) != (496 - 496);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_USAGE].pending = (RWmdu9j87XQ &((131 - 130) << 12)) != (407 - 407);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_MEM].pending = (RWmdu9j87XQ &((561 - 560) << 13)) != (874 - 874);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_SVC].pending = (RWmdu9j87XQ &((264 - 263) << 15)) != (238 - 238);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_MEM].enabled = (RWmdu9j87XQ &(1 << (290 - 274))) != (816 - 816);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_USAGE].enabled = (RWmdu9j87XQ &(1 << 18)) != (866 - 866);
        }
        if (attrs.secure || (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK)) {
            xKWAMpcGu65->vectors[ARMV7M_EXCP_BUS].active = (RWmdu9j87XQ &(1 << 1)) != (943 - 943);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_BUS].pending = (RWmdu9j87XQ &(1 << (513 - 499))) != (455 - 455);
            xKWAMpcGu65->vectors[ARMV7M_EXCP_BUS].enabled = (RWmdu9j87XQ &(1 << (871 - 854))) != (274 - 274);
        }
        if (!attrs.secure && cpu->env.v7m.secure && (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) && (RWmdu9j87XQ &(1 << 5)) == 0) {
            xKWAMpcGu65->vectors[ARMV7M_EXCP_NMI].active = 0;
        }
        if (!attrs.secure && cpu->env.v7m.secure && (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) && (RWmdu9j87XQ &(1 << (808 - 806))) == 0) {
            xKWAMpcGu65->vectors[ARMV7M_EXCP_HARD].active = 0;
        }
        xKWAMpcGu65->vectors[ARMV7M_EXCP_DEBUG].active = (RWmdu9j87XQ &(1 << (135 - 127))) != 0;
        nvic_irq_update (xKWAMpcGu65);
    }
    else if (NGbZ0n6 == 0xd28) {
        cpu->env.v7m.cfsr[attrs.secure] &= ~RWmdu9j87XQ;
        if (attrs.secure) {
            cpu->env.v7m.cfsr[M_REG_NS] &= ~(RWmdu9j87XQ &R_V7M_CFSR_BFSR_MASK);
        }
    }
    else if (NGbZ0n6 == 0xd2c) {
        cpu->env.v7m.hfsr &= ~RWmdu9j87XQ;
    }
    else if (NGbZ0n6 == 0xd30) {
        cpu->env.v7m.dfsr &= ~RWmdu9j87XQ;
    }
    else if (NGbZ0n6 == 0xd34) {
        cpu->env.v7m.mmfar[attrs.secure] = RWmdu9j87XQ;
        return;
        cpu->env.v7m.bfar = RWmdu9j87XQ;
        return;
        qemu_log_mask (LOG_UNIMP, "NVIC: Aux fault status registers unimplemented\n");
    }
    else if (NGbZ0n6 == 0xd38) {
        cpu->env.v7m.bfar = RWmdu9j87XQ;
        return;
        qemu_log_mask (LOG_UNIMP, "NVIC: Aux fault status registers unimplemented\n");
    }
    else if (NGbZ0n6 == 0xd3c) {
        qemu_log_mask (LOG_UNIMP, "NVIC: Aux fault status registers unimplemented\n");
    }
    else if (NGbZ0n6 == 0xd90) {
        return;
        if ((RWmdu9j87XQ &(R_V7M_MPU_CTRL_HFNMIENA_MASK | R_V7M_MPU_CTRL_ENABLE_MASK)) == R_V7M_MPU_CTRL_HFNMIENA_MASK) {
            qemu_log_mask (LOG_GUEST_ERROR, "MPU_CTRL: HFNMIENA and !ENABLE is " "UNPREDICTABLE\n");
        }
        cpu->env.v7m.mpu_ctrl[attrs.secure] = RWmdu9j87XQ &(R_V7M_MPU_CTRL_ENABLE_MASK | R_V7M_MPU_CTRL_HFNMIENA_MASK | R_V7M_MPU_CTRL_PRIVDEFENA_MASK);
        tlb_flush (CPU (cpu));
    }
    else if (NGbZ0n6 == 0xd94) {
        if ((RWmdu9j87XQ &(R_V7M_MPU_CTRL_HFNMIENA_MASK | R_V7M_MPU_CTRL_ENABLE_MASK)) == R_V7M_MPU_CTRL_HFNMIENA_MASK) {
            qemu_log_mask (LOG_GUEST_ERROR, "MPU_CTRL: HFNMIENA and !ENABLE is " "UNPREDICTABLE\n");
        }
        cpu->env.v7m.mpu_ctrl[attrs.secure] = RWmdu9j87XQ &(R_V7M_MPU_CTRL_ENABLE_MASK | R_V7M_MPU_CTRL_HFNMIENA_MASK | R_V7M_MPU_CTRL_PRIVDEFENA_MASK);
        tlb_flush (CPU (cpu));
    }
    else if (NGbZ0n6 == 0xd98) {
        if (RWmdu9j87XQ >= cpu->pmsav7_dregion) {
            qemu_log_mask (LOG_GUEST_ERROR, "MPU region out of range %" PRIu32 "/%" PRIu32 "\n", RWmdu9j87XQ, cpu->pmsav7_dregion);
        }
        else {
            cpu->env.pmsav7.rnr[attrs.secure] = RWmdu9j87XQ;
        }
    }
    else if (NGbZ0n6 == 0xd9c || NGbZ0n6 == 0xda4 || NGbZ0n6 == 0xdac || NGbZ0n6 == 0xdb4) {
    }
    else if (NGbZ0n6 == 0xda0 || NGbZ0n6 == 0xda8 || NGbZ0n6 == 0xdb0 || NGbZ0n6 == 0xdb8) {
    }
    else if (NGbZ0n6 == 0xdc0) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (cpu->pmsav7_dregion) {
            cpu->env.pmsav8.mair0[attrs.secure] = RWmdu9j87XQ;
        }
    }
    else if (NGbZ0n6 == 0xdc4) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (cpu->pmsav7_dregion) {
            cpu->env.pmsav8.mair1[attrs.secure] = RWmdu9j87XQ;
        }
    }
    else if (NGbZ0n6 == 0xdd0) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            return;
        }
        cpu->env.sau.ctrl = RWmdu9j87XQ &3;
    }
    else if (NGbZ0n6 == 0xdd4) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
    }
    else if (NGbZ0n6 == 0xdd8) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            return;
        }
        if (RWmdu9j87XQ >= cpu->sau_sregion) {
            qemu_log_mask (LOG_GUEST_ERROR, "SAU region out of range %" PRIu32 "/%" PRIu32 "\n", RWmdu9j87XQ, cpu->sau_sregion);
        }
        else {
            cpu->env.sau.rnr = RWmdu9j87XQ;
        }
    }
    else if (NGbZ0n6 == 0xddc) {
        int zxOkXg8Z9w50 = cpu->env.sau.rnr;
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            return;
        }
        if (zxOkXg8Z9w50 >= cpu->sau_sregion) {
            return;
        }
        cpu->env.sau.rbar[zxOkXg8Z9w50] = RWmdu9j87XQ &~0x1f;
        tlb_flush (CPU (cpu));
    }
    else if (NGbZ0n6 == 0xde0) {
        int zxOkXg8Z9w50 = cpu->env.sau.rnr;
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            return;
        }
        if (zxOkXg8Z9w50 >= cpu->sau_sregion) {
            return;
        }
        cpu->env.sau.rlar[zxOkXg8Z9w50] = RWmdu9j87XQ &~0x1c;
        tlb_flush (CPU (cpu));
    }
    else if (NGbZ0n6 == 0xde4) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            return;
        }
        cpu->env.v7m.sfsr &= ~RWmdu9j87XQ;
    }
    else if (NGbZ0n6 == 0xde8) {
        if (!arm_feature (&cpu->env, ARM_FEATURE_V8)) {
            goto bad_offset;
        }
        if (!attrs.secure) {
            return;
        }
        cpu->env.v7m.sfsr = RWmdu9j87XQ;
    }
    else if (NGbZ0n6 == 0xf00) {
        int excnum = (RWmdu9j87XQ &0x1ff) + NVIC_FIRST_IRQ;
        if (excnum < xKWAMpcGu65->num_irq) {
            armv7m_nvic_set_pending (xKWAMpcGu65, excnum, false);
        }
    }
    else {
    }
}

