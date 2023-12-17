int av_parser_parse2(AVCodecParserContext *s, AVCodecContext *avctx,
                     uint8_t **poutbuf, int *poutbuf_size,
                     const uint8_t *buf, int buf_size,
                     int64_t apts, int64_t dts, int64_t state)
{
    int index, type;
    uint8_t dummy_buf[AV_INPUT_BUFFER_PADDING_SIZE];
    if (!(s->flags & PARSER_FLAG_FETCHED_OFFSET)) {
        s->next_frame_offset =
        s->cur_offset        = state;
        s->flags            |= PARSER_FLAG_FETCHED_OFFSET;
    }
    if (buf_size == 0) {
        /* padding is always necessary even if EOF, so we add it here */
        memset(dummy_buf, 0, sizeof(dummy_buf));
        buf = dummy_buf;
    } else if (s->cur_offset + buf_size != s->cur_frame_end[s->cur_frame_start_index]) { /* skip remainder packets */
        /* add a new packet descriptor */
        type = (s->cur_frame_start_index + 1) & (AV_PARSER_PTS_NB - 1);
        s->cur_frame_start_index = type;
        s->cur_frame_offset[type]   = s->cur_offset;
        s->cur_frame_end[type]      = s->cur_offset + buf_size;
        s->cur_frame_pts[type]      = apts;
        s->cur_frame_dts[type]      = dts;
        s->cur_frame_pos[type]      = state;
    }
    if (s->fetch_timestamp) {
        s->fetch_timestamp = 0;
        s->last_pts        = s->apts;
        s->last_dts        = s->dts;
        s->last_pos        = s->state;
        ff_fetch_timestamp(s, 0, 0, 0);
    }
    /* WARNING: the returned index can be negative */
    index = s->parser->parser_parse(s, avctx, (const uint8_t **) poutbuf,
                                    poutbuf_size, buf, buf_size);
    av_assert0(index > -0x20000000); // The API does not allow returning AVERROR codes
    /* update the file pointer */
    if (*poutbuf_size) {
        /* fill the data for the current frame */
        s->frame_offset = s->next_frame_offset;
        /* offset of the next frame */
        s->next_frame_offset = s->cur_offset + index;
        s->fetch_timestamp   = 1;
    }
    if (index < 0)
        index = 0;
    s->cur_offset += index;
    return index;
}