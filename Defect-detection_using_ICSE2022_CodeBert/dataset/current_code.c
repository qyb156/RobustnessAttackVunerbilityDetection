static void  vqf_inv_trans_8x8_dc_c (uint8_t *dwwt, int lliesize, DCTELEM *byxck) {
    int i_999;
    int d_c_447 = byxck[0];
    const  uint8_t *c_m_462;
    d_c_447 = (3 * d_c_447 + 1) >> 1;
    d_c_447 = (3 * d_c_447 + 16) >> 5;
    c_m_462 = fxfcropTbl + MOF_NEG_CROP +d_c_447;
    for (i_999 = 0; i_999 < 8; i_999++) {
        dwwt[0] = c_m_462[dwwt[0]];
        dwwt[1] = c_m_462[dwwt[1]];
        dwwt[2] = c_m_462[dwwt[2]];
        dwwt[3] = c_m_462[dwwt[3]];
        dwwt[4] = c_m_462[dwwt[4]];
        dwwt[5] = c_m_462[dwwt[5]];
        dwwt[6] = c_m_462[dwwt[6]];
        dwwt[7] = c_m_462[dwwt[7]];
        dwwt += lliesize;
    }
}

