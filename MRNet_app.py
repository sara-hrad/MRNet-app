from util import *
import streamlit as st
import cv2
from PIL import Image

##############################################################################
###################           streamlit-page title      ######################
##############################################################################
apptitle = 'SpinsightMRI'
st.set_page_config(
    page_title=apptitle,
    page_icon=Image.open('./data/nobackicon.png'),
    layout="centered",
    initial_sidebar_state = "auto")

st.title('Spinsight - Adding Insight to Injury')

st.sidebar.image('./data/nobackicon.png', caption="Spinsight")
st.sidebar.markdown("## Select Patient Information")

##############################################################################
###################           Loading Data              ######################
##############################################################################
train_acl = pd.read_csv('./data/MRNet-v1.0/train-acl.csv', header=None,
                       names=['Case', 'Abnormal'], 
                       dtype={'Case': str, 'Abnormal': np.int64})

train_acl.head()
print(train_acl.shape)
train_acl.Abnormal.value_counts(normalize=True)


##############################################################################
####################     One stack/case visualization   ######################
##############################################################################
cases = load_cases(n=100)
slice_nums = {}
for case in cases:
    slice_nums[case] = {}
    for plane in ['coronal', 'sagittal', 'axial']:
        slice_nums[case][plane] = cases[case][plane].shape[0]

gt = np.load('./data/results/labels.npy')
p_plane = np.load('./data/results/predictionperplane.npy')
p_patient = np.load('./data/results/predperpatiant.npy')

P_ID = st.sidebar.selectbox('Specify the patient number:', list(cases.keys()))
task = st.sidebar.selectbox('Task:', ['Overall Insight', 'Data Visualization', 'ACL Tear', 'Cardio Vascular', 'Radiologist Report'])

if task == 'ACL Tear':
    plane = st.sidebar.multiselect('MRI Plane:', ['sagittal', 'coronal', 'axial'])

    if plane:
        st.subheader('Raw data')
        col1, col2, col3 = st.columns([1,1,1])
        if 'sagittal' in plane:
            sag_bar = st.sidebar.slider('Sagittal', min_value=0, max_value=slice_nums[P_ID]['sagittal'] - 1  , step=1)
            col1.image(cases[P_ID]['sagittal'][sag_bar], caption = f'MRI slice {sag_bar} on sagittal plane')
        if 'coronal' in plane:
            cor_bar = st.sidebar.slider('Coronal', min_value=0, max_value=slice_nums[P_ID]['coronal'] - 1  , step=1)
            col2.image(cases[P_ID]['coronal'][cor_bar], caption = f'MRI slice {cor_bar} on coronal plane')
        if 'axial' in plane:
            ax_bar = st.sidebar.slider('Axial', min_value=0, max_value=slice_nums[P_ID]['axial'] - 1  , step=1)
            col3.image(cases[P_ID]['axial'][ax_bar], caption = f'MRI slice {ax_bar} on axial plane')
        
        st.subheader('Scan information')
        st.info(f"""
        **Ground truth :** {gt[list(cases.keys()).index(P_ID)]}\n
        **Model prediction :** {p_patient[list(cases.keys()).index(P_ID)]*100:.2f} % ACL tear
        """)
        with st.expander("What is ground truth & model prediction?"):
            st.write("""
            **Ground truth** is the actual label used for training, usually this information is provided 
            from radiologist's diagnosis. \n
            **Model prediction** is out put of model after training with labeled data. The model now can 
            predict the diagnosis of an unseen datapoint.
            """)
        with st.expander("What is model performance metric?"):
            col11, col22 = st.columns([2,1])
            col11.write("""
            Here we used **area under ROC curve (AUC)**:
            
            Model AUC: 85.35 % (rank 6 based on the competition [leader board](https://stanfordmlgroup.github.io/competitions/mrnet/).)
            
            The value of AUC characterizes the model performance. 
            Higher the AUC value, higher the performance of the model. The perfect classifier will have high 
            value of true positive rate and low value of false positive rate.
            """)
            auc_img = Image.open('./data/auc.png')
            col22.image(auc_img, caption = f'ROC curve')

    ##############################################################################
    ####################     dataset information            ######################
    ##############################################################################
    dataset = st.sidebar.checkbox('About dataset')
    if dataset:
        st.subheader('MRNet dataset')
        st.info("""
        MRNet is the knee MRI dataset provided by Stanford. It's splitted into:

        - **training set :** 1130 cases
        - **validation set :** 120 cases
        - **test set :** 120 cases

        **Note :** We don't have access to the test set.

        Each case has maximum of three MRI plane; axial, coronal, and sagital. There are .csv files to indicate the ground truth/labels for three different tasks; ACL tear, meniscus and abnormality.\n
        The dataset page and competition results are available 
        [here](https://stanfordmlgroup.github.io/competitions/mrnet/).
        """)
    ##############################################################################
    ####################     model information              ######################
    ##############################################################################
    show_model = st.sidebar.checkbox('Show model used for training')
    image = cv2.imread('./data/overview.png')

    if show_model:
        st.subheader('Model Structure')
        st.image(image, caption ='Model overview.', channels='BGR')
        st.info("""
        We trained three independent CNN models on data from each plane. Each of these models are now specialize in detecting ACL tear from a given plane.
        
        We integrated all these models by training a logistic regression on the outputs of independent ACL tear clasifiers.
        
        **This suposed to mimic the way radiologists consider different MRI scans in different planes of a single patient in order to make the final diagnostic.**
        """)


if task == 'Cardio Vascular':
    st.subheader('Cardiovascular Risk Assessment')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<h1 style='text-align: center;font-size: 20px;'>Original Image</h1>", unsafe_allow_html=True)
        image_1 = Image.open('./data/original_scan_how_it_work.PNG')
        st.image(image_1, caption='Axial PD')
    with col2:
        st.markdown("<h1 style='text-align: center;font-size: 20px;'>Artery Detection</h1>", unsafe_allow_html=True)
        image_2= Image.open('./data/artery_detection_how_it_work.PNG')
        st.image(image_2, caption = 'Artery vessel is detected')
    with col3:
        st.markdown("<h1 style='text-align: center;font-size: 20px;'>Wall Segmentation</h1>", unsafe_allow_html=True)
        image_2= Image.open('./data/calcified_plaque_flow_artifact_wall_segmentation.PNG')
        st.image(image_2, caption='Inner (red) and outer (green) artery vessel wall')
    with col4:
        st.markdown("<h1 style='text-align: center;font-size: 20px;'>Feature Calculation</h1>", unsafe_allow_html=True)
        image_2= Image.open('./data/feature_calculation_how_it_work.PNG')
        st.image(image_2, caption='The dimensions of artery vessel wall')
    with col5:
        st.markdown("<h1 style='text-align: center;font-size: 20px;'>Abnormalities Detection</h1>", unsafe_allow_html=True)
        image_2= Image.open('./data/calcified_plaque_flow_artifact.PNG')
        st.image(image_2, caption='Calcified plaque (red arrow) and severe flow artifacts (green)')

    st.markdown("<h1 style='text-align: left;font-size: 20px;'>AI Report</h1>", unsafe_allow_html=True)
    st.markdown("""
    >A clacified plaque and severe flow artifacts are detected in artery vessels. This may be a 
    sign of atherosclerosis. The report is sent to your family doctor, and they will advise you 
    regarding this condition and any course of treatment.
    """)
    Tech_exp = st.sidebar.checkbox('The risk assessment model')

    if Tech_exp:
        with st.container():
            st.markdown("<h1 style='text-align: left;font-size: 20px;'>Cardiovascular Risk Assessment Model Description</h1>", unsafe_allow_html=True)
            st.markdown("""
            >A fully automated and robust analysis technique for popliteal artery evaluation (FRAPPE) is developed 
            using innovative machine learning techniques, including object tracking and a 256-layers deep neural 
            network to detect the vessel wall, its dimension, and any plaque.
            """)
            st.markdown(""">The performance of FRAPPE in the Osteoarthritis Initiative (OAI) dataset is validated by 
            comparing its measurements with manual measurements and estimating scan-rescan repeatability. 
            A preliminary assessment of FRAPPE’s ability to discriminate between diseased and non-diseased arteries 
            is performed by comparing FRAPPE-based measurements between subjects with high and low cardiovascular risk.
            """)

    
    #--side bar
    st.sidebar.markdown('## Patient Information:')
    st.sidebar.info("""
    **Name:** Doe, Regina \n
    **Date of birth:** 1984/01/15\n
    **Referring physician:** Banner, Ross, MD
    """)
    st.sidebar.selectbox('Exam date:',['2021/11/30', '2010/05/06'])


if task == 'Overall Insight':
    st.subheader('Overall Insight')
    st.markdown("""
    > Regina, your MRI results indicate a tear in your anterior cruciate ligament (ACL), 
    an injury that typically occur when a person is running or jumping and then suddenly slows and changes 
    direction (eg, cutting) or pivots in a way that involves rotating or bending the knee sideways.
    
    > Women appear to be at a higher risk of non-contact ACL injuries than men, although the exact reason for this is not clear.

    Based on your Spinsight results, we recommend the following:
    """)
    WB = st.checkbox('Weight Bearing')
    exercise = st.checkbox('Exercise Recommendation')
    Brace = st.checkbox('Brace Recommendation')
    
    if WB:
        st.markdown("<h1 style='text-align: left;font-size: 20px;'>Weight Bearing Considration</h1>", unsafe_allow_html=True)
        st.markdown("""> You can continue weight bearing, but be cautious of movements such as squatting, pivoting,
            and stepping sideways, and activities such as walking down stairs, in which the entire body weight is placed
            on the affected leg. These movements may cause a feeling of unsteadiness.
            """) 

    if exercise:
        st.markdown("<h1 style='text-align: left;font-size: 20px;'>Exercise Recommendation</h1>", unsafe_allow_html=True)
        st.markdown("""> To help the patient overcome the problem of a torn ligament in knee, here are knee ligament injury 
        treatment exercises to recommend. These exercises are most effective when performed regularly, 3-4 times a day and
        in short intervals of few times every hour rather than once a day for a longer period.
        """)
        with st.expander("Bridge"):
            st.write("""
            Lie on your back. Bend both your knees and extend both your arms along your body, palms face down. 
            Lift your pelvis off the ground and puff your chest towards your chin. 
            Now, start rolling your thighs inward and down. 
            Remember to keep your knees stacked on your ankles throughout the duration of the exercise. 
            Hold for 10 secs and repeat 1 set of ten reps on each side.
        """)
            st.image("./data/bridges.PNG")
    
    if Brace:
        st.markdown("<h1 style='text-align: left;font-size: 20px;'>Brace Recommendation</h1>", unsafe_allow_html=True)
        st.markdown("""
        > We recommend use of a wrap, hinged knee brace which can improve function and may allow for a speedier return to activity or sport. 

        > We recommend a “DonJoy Deluxe Hinged Knee Brace” in size medium (fitted from your MRI results). This knee brace combines bilateral
         hinges with a breathable compression fabric designed to minimize skin irritation and discomfort.
        """)
        st.markdown("Brace style: Wrap, hinged")
        link = '[The DonJoy Deluxe Hinged Knee Brace](https://www.amazon.ca/dp/B00P3D1HWE?ref_=as_li_ss_tl&language=en_US&correlationId=e54ab87c-26b9-46f1-ab02-7f3023d82123&linkCode=gs2&linkId=b6f01cb2813a2601245905d02ca6d3ad&tag=healthlineca-20)'
        st.markdown('We recommend: ') 
        st.markdown(link, unsafe_allow_html=True)
        with st.expander("What are the pros and cons of brace?"):
            st.markdown(">Pro: Can improve function and may allow for a speedier return to activity or sport.")
            st.markdown(">Con: If a brace significantly restricts movement, muscle atrophy and stiffness may occur. This knee brace combines bilateral hinges with a breathable compression fabric designed to minimize skin irritation and discomfort.")
    
    #--side bar
    st.sidebar.markdown('## Patient Information:')
    st.sidebar.info("""
    **Name:** Doe, Regina \n
    **Date of birth:** 1984/01/15\n
    **Referring physician:** Banner, Ross, MD
    """)
    st.sidebar.selectbox('Exam date:',['2021/11/30', '2010/05/06'])


if task == 'Radiologist Report':
    st.subheader('MRI left knee - sample report')
    st.write("""
    >**Indication :** Left knee twisting injury on 1/1/2013.

    >**Technique:** Magnetic resonance imaging of the left knee joint is submitted with standard protocol,
    sagittal T1/T2,coronal PD, TI and stir images, axial T2 and/or GE sequences having been acquired in the
    dedicated knee joint coil without IV contrast. Exam performed on .3 Tesla Open MRI system
    """)
    st.markdown("<h1 style='text-align: left;font-size: 20px;'>FINDINGS:</h1>", unsafe_allow_html=True)
    st.info("""
    Extensor mechanism: The extensor mechanism is intact. The patellar ligament is intact.
    - **Ligaments:** There is complete mid substance disruption of the anterior cruciate ligament. Its fibers fibrillate
    within the joint. The posterior crucial ligament is buckled as a result. Medial collateral ligament: There is
    edema relative to the MCL, external to such on the basis of grade 1 injury.
    - **Lateral collateral ligament complex:** The iliotibial band, biceps femoris tendon, fibular collateral ligament
    and popliteus muscle and tendon are thought to be intact.
    - **Menisci:** The anterior and posterior horns of the lateral meniscal tissue are intact. The transverse
    geniculate ligament is indistinct. The anterior and posterior horns of the medial meniscal tissue are intact
    though there is posterior meniscal capsular junction edema.
    - **Patella:** The retropatellar cartilage is preserved and the patellar retinacula normal in appearance.
    - **Articulation:** There is a large suprapatellar bursal effusion. Mild reactive synovitis exists.
    - **Osseous structures:** There is a deepened lateral condylar patellar sulcus of the femur. Bone cortical and
    marrow signal intensity is otherwise unremarkable.
    - **Periarticular soft tissues:** Regional edema negative for popliteal fossa cyst or ganglion
    """)
    st.markdown("<h1 style='text-align: left;font-size: 20px;'>Impression:</h1>", unsafe_allow_html=True)
    st.info("""
    - Complete full-thickness disruption of the anterior cruciate ligament.
    -  Associated osseous contusion of the lateral condylar patellar sulcus: Pivot shift injury.
    - Grade 1 MCL complex injury. 
    - No other associated injury identified
    """)
    
    #--side bar
    st.sidebar.markdown('## Patient Information:')
    st.sidebar.info("""
    **Name:** Doe, Regina \n
    **Date of birth:** 1984/01/15\n
    **Referring physician:** Banner, Ross, MD
    """)
    st.sidebar.selectbox('Exam date:',['2021/11/30', '2010/05/06'])


if task == 'Data Visualization':
    plane = st.sidebar.multiselect('MRI Plane:', ['sagittal', 'coronal', 'axial'])

    if plane:
        st.subheader('Raw data')
        col1, col2, col3 = st.columns([1,1,1])
        if 'sagittal' in plane:
            sag_bar = st.sidebar.slider('Sagittal', min_value=0, max_value=slice_nums[P_ID]['sagittal'] - 1  , step=1)
            col1.image(cases[P_ID]['sagittal'][sag_bar], caption = f'MRI slice {sag_bar} on sagittal plane')
        if 'coronal' in plane:
            cor_bar = st.sidebar.slider('Coronal', min_value=0, max_value=slice_nums[P_ID]['coronal'] - 1  , step=1)
            col2.image(cases[P_ID]['coronal'][cor_bar], caption = f'MRI slice {cor_bar} on coronal plane')
        if 'axial' in plane:
            ax_bar = st.sidebar.slider('Axial', min_value=0, max_value=slice_nums[P_ID]['axial'] - 1  , step=1)
            col3.image(cases[P_ID]['axial'][ax_bar], caption = f'MRI slice {ax_bar} on axial plane')
        

    ##############################################################################
    ####################     dataset information            ######################
    ##############################################################################
    dataset = st.sidebar.checkbox('About dataset')
    if dataset:
        st.subheader('MRNet dataset')
        st.info("""
        MRNet is the knee MRI dataset provided by Stanford. It's splitted in:

        - **training set :** 1130 cases
        - **validation set :** 120 cases
        - **test set :** 120 cases

        **Note :** We don't have access to the test set.

        Each case has maximum of three MRI plane; axial, coronal, and sagital. There are .csv files to indicate the ground truth/labels for three different tasks; ACL tear, meniscus and abnormality.\n
        The dataset page and competition results are availabel 
        [here](https://stanfordmlgroup.github.io/competitions/mrnet/).
        """)

link = '[github](https://github.com/sara-hrad/MRNet-app)'
st.sidebar.markdown(link, unsafe_allow_html=True)