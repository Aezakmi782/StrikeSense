import streamlit
import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import sklearn
import streamlit.components.v1 as components
import plotly.graph_objects as go
import webbrowser

st.session_state.fav_text = "Favourite\nclick predict"
st.session_state.ud_text = "Underdog\nclick predict"

st.set_page_config(layout="wide")

st.markdown("<style> .css-18ni7ap{ visibility: hidden; } div.block-container {padding-top:1rem;} </style>",
            unsafe_allow_html=True)

df = pd.read_csv("fights_to_analyze.csv")


# prediction

def form(fighter_name, datum):
    fights_df = pickle.load(open("fights_df.pkl", 'rb'))
    vysledek = ''
    skore = 0
    koef = 0.1
    result = ['W' if x == 1 else 'L' for x in
              fights_df['result'][(fights_df['fighter'] == fighter_name) & (fights_df['date'] < datum)]]
    for vyhra in result[:-6:-1]:
        if vyhra == 'W':
            skore += koef
        else:
            skore -= koef
        koef += 0.1
        vysledek += vyhra + ' '
    vysledek = vysledek[:-1]
    return vysledek, skore


def head_to_head(_arg1, _arg2):
    svc = pickle.load(open("model.pkl", 'rb'))
    fighters_to_analyze = pickle.load(open("fighters_to_analyze.pkl", 'rb'))
    form_fighter = form(_arg1, "2022-12-12")
    form_opponent = form(_arg2, "2022-12-12")
    h1 = fighters_to_analyze[fighters_to_analyze.fighter == _arg1].copy()
    h1.loc[:, 'form_skore'] = form_fighter[1]
    h2 = fighters_to_analyze[fighters_to_analyze.fighter == _arg2].copy()
    h2.loc[:, 'form_skore'] = form_opponent[1]
    h1.loc[:, "opponent"] = _arg2
    h1 = h1.merge(h2, left_on="opponent", right_on="fighter", how="inner", suffixes=("_fighter", "_opponent"))
    h1 = h1.loc[:,
         ["ground_def_skill_fighter", "ground_att_skill_fighter", "stand_att_skill_fighter", "stand_def_skill_fighter",
          "stamina_fighter", "form_skore_fighter",
          "ground_def_skill_opponent", "ground_att_skill_opponent", "stand_att_skill_opponent",
          "stand_def_skill_opponent", "stamina_opponent", "form_skore_opponent"]]
    probs = svc.predict_proba(h1)
    prob_fighter = probs[0][1]
    prob_opponent = probs[0][0]
    return [prob_fighter, prob_opponent]


# read csv

fights_df = pd.read_csv('fights_df.csv', parse_dates=True)
fighters_df = pd.read_csv('fighters_df.csv')
ufc_fighters = pd.DataFrame(fights_df.drop_duplicates("fighter")["fighter"])

col_y = ['Ground Defence', 'Stand Defence', 'Stand Attack', 'Stamina']

df.drop(columns=['Unnamed: 0', 'date', 'result', 'method', 'opponent'], inplace=True)

df.rename(columns={
    'ground_def_skill_fighter': 'Ground Defence',
    'stand_def_skill_fighter': 'Stand Defence',
    'stand_att_skill_fighter': 'Stand Attack',
    'stamina_fighter': 'Stamina'
}, inplace=True)


def normalize(df):
    result = df.copy()
    for feature_name in col_y:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df_normalize = normalize(df)


def update_graph(f1, f2):
    f1_x = df_normalize[df_normalize['fighter_fighter'] == f1].iloc[0, :].values.tolist()[1:]
    f2_x = df_normalize[df_normalize['fighter_fighter'] == f2].iloc[0, :].values.tolist()[1:]

    trace1 = go.Bar(
        y=col_y,
        x=[x * -1 for x in f1_x],
        name=f1,
        orientation='h',
        hoverinfo='none',
        marker=dict(
            color='rgba(255, 80, 81, 0.8)',
            line=dict(
                color='rgba(255, 80, 81, 1.0)',
                width=3)
        )
    )
    trace2 = go.Bar(
        y=col_y,
        x=f2_x,
        name=f2,
        orientation='h',
        hoverinfo='none',
        marker=dict(
            color='rgba(70, 179, 120, 0.8)',
            line=dict(
                color='rgba(70, 179, 120, 1.0)',
                width=3)
        )
    )

    return {

        'data': [trace1, trace2],
        'layout': go.Layout(
            barmode='overlay',
            title='Fight Stats',
            title_x=0.45,
            titlefont={
                'size': 30
            },
            paper_bgcolor='rgba(0,0,0,0.7)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title_font_color='white',
            xaxis=dict(
                range=[-1, 1],
                showticklabels=False
            )
        )

    }


st.markdown("<h1 style='text-align: center; color: white; font-size: 80px;'>Strike Sense</h1>", unsafe_allow_html=True)
# st.title(
#     "Check out players stats [link](https://public.tableau.com/views/StrikeSense/StrikeSenceStatsOverView?:language"
#     "=en-US&publish=yes&:display_count=n&:origin=viz_share_link"
#     "&:display_count=n&:origin=viz_share_link)")
st.markdown("""   <style>
   .stApp {
   background-image: url('https://girraphic.com/wp-content/uploads/2020/07/Screen-Shot-2020-07-01-at-11.22.07-am.png');
   background-size: cover;
   }
   </style>   """, unsafe_allow_html=True)

left = Image.open('fighter_left.png')
right = Image.open('fighter_right.png')

# columns for selection

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    st.markdown("<h2 style='text-align: center; color: white;'>Fighter 1</h2>",
                unsafe_allow_html=True)

    fav_fighter = st.selectbox("Select Fighter", df['fighter_fighter'].unique())

with c3:
    st.markdown("<h2 style='text-align: center; color: white;'>Fighter 2</h2>",
                unsafe_allow_html=True)

    ud_fighter = st.selectbox("Select Fighter", df['fighter_fighter'].unique(), key="underdog fighter")

with c2:
    st.plotly_chart(update_graph(fav_fighter, ud_fighter), use_container_width=True)

st.markdown("<style> .css-10trblm {text-align: center; }</style>", unsafe_allow_html=True)


# button function
def btn_click():
    prob_list = head_to_head(fav_fighter, ud_fighter)
    st.session_state.fav_text = f"{round(prob_list[0] * 100, 2)}%"
    st.session_state.ud_text = f"{round(prob_list[1] * 100, 2)}%"
    c04.markdown('<h1 style="color:white;font-size:24px;">Chances of Winning</h1>', unsafe_allow_html=True)
    c02.markdown('<h1 style="color:white;font-size:24px;">Chances of Winning</h1>', unsafe_allow_html=True)
    if prob_list[0] > prob_list[1]:
        c04.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{st.session_state.fav_text}</h1>',
                     unsafe_allow_html=True)
        c02.markdown(f'<h1 style="color:red;font-size:24px;">{st.session_state.ud_text}</h1>', unsafe_allow_html=True)

    else:
        c04.markdown(f'<h1 style="color:red;font-size:24px;">{st.session_state.fav_text}</h1>', unsafe_allow_html=True)
        c02.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{st.session_state.ud_text}</h1>',
                     unsafe_allow_html=True)


col1, col2, col3 = st.columns((1, 1, 2))

# predict button
with col3:
    predict_btn = st.button("Predict", on_click=btn_click)
    st.markdown('<style> .css-1n543e5{ background-color: red; padding: 1rem 3rem; color:white;} </style>',
                unsafe_allow_html=True)

c01, c02, c03, c04, c05 = st.columns([1, 2, 2, 2, 1])

# st.title(
#     "Check out players stats [link](https://public.tableau.com/shared/DWDC2BYDK?:display_count=n&:origin=viz_share_link"
#     "=en-US&publish=yes&:display_count=n&:origin=viz_share_link"
#     "&:display_count=n&:origin=viz_share_link)")


with c01:
    st.image(left, use_column_width=True)

with c02:
    st.title('#')
    st.title('#')

with c04:
    st.title('#')
    st.title('#')

with c05:
    st.image(right, use_column_width=True)

col1, col2, col3 = st.columns([0.69, 0.2, 0.5])

with col2:
    url = """ <style>
    a:link, a:visited {
      background-color: rgb(14, 17, 23);
      color: white;
      padding: 8px 18px;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      border-radius: 10px;
    }

    a:hover, a:active {
      color: red;
      border-color: red;
      }
    </style>

    <a href="https://public.tableau.com/app/profile/vishal.rohila/viz/StrikeSense_16886442232640/StrikeSenseStatsOverView?publish=yes">Players Stats</a>
    """

    st.markdown(url, unsafe_allow_html=True)