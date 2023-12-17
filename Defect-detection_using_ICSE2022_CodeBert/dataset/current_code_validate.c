static int sLuPfEZx7lGq (const  char *SfH7Mv4m) {
    OADTh5vu *Z9Ah0u;
    int lrHckXOzV;
    Z9Ah0u = qmp ("{ 'execute': 'qom-get', 'arguments': { 'path': '%s', " "'property': 'temperature' } }", SfH7Mv4m);
    g_assert (qdict_haskey (Z9Ah0u, "return"));
    lrHckXOzV = qdict_get_int (Z9Ah0u, "return");
    QDECREF (Z9Ah0u);
    return lrHckXOzV;
}

