import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

aws_api_url = "https://ky9x9gezmj.execute-api.eu-west-3.amazonaws.com/v1"
aws_api_headers = {'Accept': 'application/json'}
aws_api_auth = HTTPBasicAuth('apikey', 'k8kLFxiyx7M9OcJXzCwi5fzaJkGYMaj1bm8zbVzf')
aws_access_key_id = "AKIAXRPAU26UJJQDJXQC"
aws_secret_access_key = "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"

tab_metrics, tab_test, tab_features, tab_train = st.tabs(["Métriques",
                                                           "Test",
                                                           "Features",
                                                           "Train"])


########################################################################################################################
# > METRIQUES
########################################################################################################################

with tab_metrics:
    st.write("")
    st.write("")
    st.markdown("## Prédictions")

    response = requests.get(aws_api_url + "/predicts",
                            json={},
                            headers=aws_api_headers, auth=aws_api_auth
                            ).json()

    if "statusCode" in response and response["statusCode"] == 200:
        import io

        df_predicts = pd.read_json(io.StringIO(response["result"]))
        df_predicts.columns = ["id","to_train", "dte_predict","to_predict","predicted","CER", "WER"]
        df_predicts = df_predicts.drop(["id","to_train"], axis=1)
        df_predicts["to_predict"] = df_predicts["to_predict"].apply(lambda x: x[:40] + " ..." if len(x) > 40
        else x)
        df_predicts["predicted"] = df_predicts["predicted"].apply(lambda x: x[:40] + " ..." if len(x) > 40
        else x)

        df_predicts["dte_heure"] = df_predicts["dte_predict"].apply(lambda x:x[:13]+"h")
        df_predicts["nb"] = 1

        st.bar_chart(data=df_predicts, x="dte_heure", y="nb", use_container_width=True)
        st.write("")
        st.write("")


        col_pie, col_metrics = st.columns(2)
        with col_pie:
            import matplotlib.pyplot as plt
            nb_train = len(df_predicts[df_predicts["CER"]>0.5])
            nb_ok = len(df_predicts) - nb_train
            plt.pie(x=[nb_ok, nb_train], labels=["OK\n(%s)" % nb_ok, "To train\n(%s" % nb_train])
            plt.legend=True
            st.pyplot(plt)

        with col_metrics:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("#### CER moy = %s %%" % round(100 * df_predicts["CER"].mean(), 2))
            st.write("#### WER moy = %s %%" % round(100 * df_predicts["WER"].mean(), 2))

        st.write("")
        st.write("")
        df_predicts = df_predicts.drop(["dte_heure","nb"], axis=1)
        st.dataframe(df_predicts, use_container_width=False)
    else:
        st.write(response)

########################################################################################################################
# > TESTS
########################################################################################################################


def verif_get_param(bar_lib, txt) :

    seuil = -1
    txt = txt + "\n   > requête : \n      [GET] %s/param" % aws_api_url
    bar_lib.text(txt)

    response = requests.get(aws_api_url + "/param", headers=aws_api_headers, auth=aws_api_auth)

    if response.ok:
        result = response.json()
        txt = txt + "\n   > code retour : %s " % result['statusCode']
        bar_lib.text(txt)
        txt = txt + "\n   > résultat :"
        bar_lib.text(txt)
        seuil = result["seuil_confiance"]
        txt = txt + "\n      seuil de confiance = %s" % seuil
        bar_lib.text(txt)
    else:
        txt = txt + "\n   > code retour : KO !"
        bar_lib.text(txt)

    return response.ok, txt, seuil


def verif_put_param(bar_lib, txt, seuil):

    txt = txt + "\n   > requête : \n      [PUT] %s/param \n      %s" % (aws_api_url, {"seuil_confiance": seuil})
    bar_lib.text(txt)

    response = requests.put(aws_api_url + "/param",
                            json={"seuil_confiance": seuil},
                            headers=aws_api_headers, auth=aws_api_auth)

    if response.ok:
        result = response.json()
        txt = txt + "\n   > code retour : %s " % result['statusCode']
        bar_lib.text(txt)
    else:
        txt = txt + "\n   > code retour : KO !"
        bar_lib.text(txt)

    return response.ok, txt


with tab_test:
    st.write("")
    st.write("")
    bTest = st.button("Lancer la procédure de test")

    if bTest:

        nb_progress = 0
        bar = st.progress(nb_progress)
        bar_lib = st.empty()
        txt = ""

        # param_get
        txt += "\n\n" + 50*"-"
        txt += "\nVérification endpoint 'GET Param'"
        txt += "\n" + 50 * "-"
        bar_lib.text(txt)

        test, txt, seuil_confiance = verif_get_param(bar_lib, txt)

        nb_progress += 12
        bar.progress(nb_progress)

        if test:

            # param_put
            txt += "\n\n" + 50 * "-"
            txt += "\nVérification endpoint 'PUT Param'"
            txt += "\n" + 50 * "-"

            txt = txt + "\n\n Sauvegarder le seuil"
            bar_lib.text(txt)

            test, txt, seuil_confiance = verif_get_param(bar_lib, txt)

            txt = txt + "\n\n Mettre le seuil à 0"
            bar_lib.text(txt)

            test, txt = verif_put_param(bar_lib, txt, 0)

            if test :

                txt = txt + "\n\n Lire le seuil pour vérification de la mise à jour"
                bar_lib.text(txt)

                test, txt, seuil = verif_get_param(bar_lib, txt)

                txt = txt + "\n\n Remettre la valeur initiale du seuil"
                bar_lib.text(txt)

                test, txt = verif_put_param(bar_lib, txt, seuil_confiance)

            nb_progress += 13
            bar.progress(nb_progress)





########################################################################################################################
# > FEATURES
########################################################################################################################

with tab_features:
    st.markdown("## Enrichissement des données")

    response = requests.get(aws_api_url + "/dataset_record",
                            json={},
                            headers=aws_api_headers, auth=aws_api_auth
                            ).json()

    if "statusCode" in response and response["statusCode"] == 200:
        import io
        df_records = pd.read_json(io.StringIO(response["result"]))
        df_records.columns = ["id","transcription","used_in_train"]
        df_records["transcription"] = df_records["transcription"].apply(lambda x : x[:30] + " ..." if len(x)>20
                                                                                    else x)
        st.dataframe(df_records, use_container_width=False)
    else:
        st.write(response)


########################################################################################################################
# > TRAIN
########################################################################################################################

with tab_train:
    st.write("")
    st.write("")
    st.button("Lancer un nouvel entrainment")
