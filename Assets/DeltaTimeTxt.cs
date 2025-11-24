using System;
using UnityEngine;
using System.Text;
using Cysharp.Text;
using TMPro;
using UnityEngine.Profiling;

[RequireComponent(typeof(TextMeshPro))]
public class DeltaTimeTxt : MonoBehaviour
{
	private Utf16ValueStringBuilder sb;
	private readonly char[] str;
	// private readonly StringBuilder sb = new (100);
	private TextMeshPro txt;

	private void Start()
	{
		sb = ZString.CreateStringBuilder(true);
		txt = GetComponent<TextMeshPro>();
	}

	private void LateUpdate()
	{
		sb.Clear();
		sb.Append(Time.deltaTime);
		sb.TryCopyTo(str, out int count);
		txt.SetText(str, 0, count);
	}

	private void OnDestroy() => sb.Dispose();
}
