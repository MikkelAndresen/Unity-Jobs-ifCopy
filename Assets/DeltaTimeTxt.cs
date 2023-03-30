using UnityEngine;
using System.Text;
using TMPro;

[RequireComponent(typeof(TextMeshPro))]
public class DeltaTimeTxt : MonoBehaviour
{
	private readonly StringBuilder sb = new StringBuilder(25);
	private TextMeshPro txt;

	private void Start()
	{
		txt = GetComponent<TextMeshPro>();
	}

	private void LateUpdate()
	{
		sb.Clear();
		sb.Append(Time.deltaTime);
		txt.text = sb.ToString();
	}
}
